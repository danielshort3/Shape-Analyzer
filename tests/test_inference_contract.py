"""Exercise both API Gateway event formats and the local direct-invocation contract."""
import base64
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import inference


def test_health_without_site_packages():
    probe = """
import json, sys
from inference import handler
for event in ({'httpMethod': 'GET', 'path': '/'}, {'requestContext': {'http': {'method': 'GET', 'path': '/health'}}}):
    response = handler(event, None)
    payload = json.loads(response['body'])
    assert response['statusCode'] == 200 and response['headers']['Cache-Control'] == 'no-store'
    assert payload['status'] == 'ok' and payload['model_loaded'] is False
    assert payload['img_size'] == 224
    assert payload['service'] == 'shape-analyzer' and len(payload['labels']) == 5
assert not {'torch', 'torchvision', 'numpy', 'PIL'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-c", probe], cwd=Path(inference.__file__).parent, check=True)


def test_training_config_remains_available_from_package():
    from shape_analyzer import cfg
    from shape_analyzer.config import cfg as training_config

    assert cfg is training_config


def encoded_image():
    buf = io.BytesIO()
    Image.new("L", (28, 28), 90).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.parametrize("canvas", [None, 64], ids=["legacy", "checkpoint"])
def test_warmup_and_prediction_execute_the_model(tmp_path, monkeypatch, canvas):
    import torch
    from shape_analyzer.model import ShapeNet
    from shape_analyzer.preprocessing import preprocess_description

    weights = tmp_path / "example.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        torch.save(ShapeNet(n_cls=5).state_dict(), weights)
    if canvas is not None:
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        weights.with_suffix(".preprocess.json").write_text(
            json.dumps(preprocess_description(canvas, inference.LABELS, digest)), encoding="utf-8"
        )
    monkeypatch.setattr(inference, "MODEL_PATH", str(weights))
    for name in ("_torch", "_model", "_preprocess"):
        monkeypatch.setattr(inference, name, None)

    warmup = {"httpMethod": "GET", "path": "/warmup/"}
    first = inference.handler(warmup, None)
    first_body = json.loads(first["body"])
    assert first["statusCode"] == 200 and first_body["model_loaded"] and first_body["warmed"]
    assert first_body["status"] == "ready" and first_body["service"] == "shape-analyzer"
    assert set(first_body["shape_confidences"]) == set(inference.LABELS)
    health = inference.handler({"httpMethod": "GET", "path": "/health"}, None)
    payload = json.loads(health["body"])
    assert payload["model_loaded"] and payload["img_size"] == (canvas or inference.IMG_SIZE)

    # The hook observes the same real model serving both warmup and prediction.
    forwards = []
    def observe_forward(_model, args, output):
        forwards.append((tuple(args[0].shape), output.detach().clone()))

    handle = inference._model.register_forward_hook(observe_forward)
    try:
        second = inference.handler({"rawPath": "/warmup", "requestContext": {"http": {"method": "POST"}}}, None)
        assert second["statusCode"] == 200 and len(forwards) == 1
        assert json.loads(second["body"])["shape_confidences"] == first_body["shape_confidences"]
        data = encoded_image()
        requests = [
            {"httpMethod": "POST", "path": "/", "body": json.dumps({"b64": data})},
            {"rawPath": "/predict", "requestContext": {"http": {"method": "POST"}}, "body": {"b64": data}},
            {"httpMethod": "POST", "path": "/classify", "isBase64Encoded": True, "body": data},
            {"body": json.dumps({"b64": data})},
        ]
        for request in requests:
            response = inference.handler(request, None)
            result = json.loads(response["body"])
            assert response["statusCode"] == 200
            shape, logits = forwards[-1]
            probabilities = torch.softmax(logits[0], dim=0)
            assert shape == (1, 1, payload["img_size"], payload["img_size"])
            winner = inference.LABELS[int(probabilities.argmax())]
            assert result["class"] == winner
            assert result["confidence"] == round(float(probabilities.max()), 4)
            assert result["shape_confidences"] == {
                label: round(float(prob), 4) for label, prob in zip(inference.LABELS, probabilities, strict=True)
            }
            assert result["duration_ms"] >= 0
        assert len(forwards) == len(requests) + 1
    finally:
        handle.remove()


@pytest.mark.parametrize(("payload", "status", "error"), [
    ({"httpMethod": "POST"}, 400, "MissingBody"),
    ({"body": "invalid JSON"}, 400, "BadJson"),
    ({"body": "null"}, 400, "BadJson"),
    ({"body": {}}, 400, "MissingImage"),
    ({"body": {"b64": 7}}, 400, "BadBase64"),
    ({"body": {"b64": "###!"}}, 400, "BadBase64"),
    ({"body": {"b64": "dGV4dA=="}}, 400, "BadImage"),
    ({"body": {"b64": "x" * (inference.MAX_REQ_BYTES + 1)}}, 413, "ImageTooLarge"),
    ({"httpMethod": "DELETE", "path": "/predict"}, 405, "MethodNotAllowed"),
    ({"httpMethod": "POST", "path": "/health", "body": {"b64": encoded_image()}}, 405, "MethodNotAllowed"),
    ({"rawPath": "/unknown", "requestContext": {"http": {"method": "POST"}}, "body": {"b64": encoded_image()}}, 404, "NotFound"),
    ({"httpMethod": "OPTIONS", "path": "/unknown"}, 404, "NotFound"),
    ({"requestContext": None, "body": {"b64": encoded_image()}}, 400, "BadEvent"),
    ({"requestContext": {"http": None}}, 400, "BadEvent"),
    ({"httpMethod": 17}, 400, "BadEvent"),
    ({"httpMethod": " ", "path": "/"}, 400, "BadEvent"),
    ({"path": ["/predict"], "body": {}}, 400, "BadEvent"),
])
def test_invalid_requests_fail_before_loading_model(payload, status, error, monkeypatch):
    def unexpected():
        pytest.fail("Invalid input attempted model initialization")
    monkeypatch.setattr(inference, "_load_model", unexpected)
    response = inference.handler(payload, None)
    assert response["statusCode"] == status and json.loads(response["body"])["error"] == error


def test_image_pixel_limit_and_preflight(monkeypatch):
    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 100)
    response = inference.handler({"body": {"b64": encoded_image()}}, None)
    assert response["statusCode"] == 413
    response = inference.handler({"httpMethod": "OPTIONS", "path": "/predict"}, None)
    assert response["statusCode"] == 204 and response["body"] == ""


def test_method_not_allowed_advertises_only_the_requested_routes_methods():
    warmup = inference.handler({"httpMethod": "DELETE", "path": "/warmup"}, None)
    predict = inference.handler({"httpMethod": "GET", "path": "/predict"}, None)
    assert warmup["statusCode"] == predict["statusCode"] == 405
    assert warmup["headers"]["Allow"] == "GET, POST, OPTIONS"
    assert predict["headers"]["Allow"] == "POST, OPTIONS"
    assert "Allow" not in inference.handler({"httpMethod": "GET", "path": "/health"}, None)["headers"]


def test_failed_model_setup_is_retryable(tmp_path, monkeypatch):
    import torch
    from shape_analyzer.model import ShapeNet

    weights = tmp_path / "model.pt"
    torch.save(ShapeNet(n_cls=5).state_dict(), weights)
    metadata = weights.with_suffix(".preprocess.json")
    metadata.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(inference, "MODEL_PATH", str(weights))
    for name in ("_torch", "_model", "_preprocess"):
        monkeypatch.setattr(inference, name, None)
    event = {"httpMethod": "GET", "path": "/warmup"}
    response = inference.handler(event, None)
    assert response["statusCode"] == 500
    assert json.loads(response["body"]) == {"error": "ServerError"} and inference._model is None
    metadata.unlink()
    assert inference.handler(event, None)["statusCode"] == 200
