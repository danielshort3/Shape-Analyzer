"""Compare the local serving entry point with the held-out training evaluation."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .config import cfg
from .evaluation import classification_report
from .helpers import _qd_array, load_npz


def archive(run, signature, role):
    paths = list(run.glob(f"*.{role}.v2-{signature[:20]}.npz"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {role} archive for the recorded split.")
    data = load_npz(paths[0])
    if data["signature"].item() != signature:
        raise ValueError("The stored archive does not match the training split.")
    return data


def confirmation_rows(spec, dev, test, count):
    """Draw new IDs from the complement of all training/validation/holdout rows."""
    if spec["source"] != "Google Quick, Draw!":
        raise ValueError("Independent confirmation uses the official bitmap arrays.")
    images, actual = [], []
    for cid, src in enumerate(spec["sources"]):
        array = _qd_array(src["class"])
        if hashlib.sha256(memoryview(array).cast("B")).hexdigest() != src["bitmap_sha256"]:
            raise ValueError("The upstream source content differs from the training run.")
        used = np.concatenate([a["source_id"][a["cls"] == cid] for a in (dev, test)])
        pool = np.setdiff1d(np.arange(len(array)), used)
        if count > len(pool):
            raise ValueError("Confirmation size exceeds the unused source population.")
        rng = np.random.default_rng([20261126, cid])
        selected = rng.choice(pool, count, replace=False)
        images.append(array[selected].reshape(-1, 28, 28))
        actual.extend([cid] * count)
    return np.concatenate(images), np.asarray(actual)


def check_endpoint(run, new_per_class=0):
    if new_per_class < 0:
        raise ValueError("Confirmation count must be nonnegative.")
    metrics = json.loads((run / "checkpoints/evaluation.json").read_text(encoding="utf-8"))
    spec = metrics["split"]
    labels = metrics["class_order"]
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dev, test = archive(run, signature, "development"), archive(run, signature, "holdout")
    images, actual = test["imgs"], test["cls"]
    if new_per_class:
        images, actual = confirmation_rows(spec, dev, test, new_per_class)
    train_idx = np.flatnonzero(dev["partition"] == 0)
    centroids = np.stack([
        dev["imgs"][train_idx[dev["cls"][train_idx] == cid]].mean(axis=0, dtype=np.float64)
        for cid in range(len(labels))
    ])
    entry = Path(__file__).resolve().parents[1] / "inference.py"
    module_spec = spec_from_file_location("_shape_local_endpoint", entry)
    serving = module_from_spec(module_spec)
    module_spec.loader.exec_module(serving)
    serving.MODEL_PATH = str((run / "checkpoints/model.pt").resolve())
    handler = serving.handler

    predictions, refs, latency = [], [], []
    for pixels in images:
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        body = json.dumps({"b64": base64.b64encode(buffer.getvalue()).decode("ascii")})
        start = perf_counter()
        response = handler({"body": body}, None)
        latency.append(1000 * (perf_counter() - start))
        if response["statusCode"] != 200:
            raise RuntimeError("Local handler failed on a prepared evaluation image.")
        result = json.loads(response["body"])
        predictions.append(labels.index(result["class"]))
        refs.append(int(np.square(centroids - pixels).mean(axis=(1, 2)).argmin()))
    endpoint = classification_report(actual, predictions, labels)
    reference = classification_report(actual, refs, labels)
    if not new_per_class and (
        endpoint["confusion_actual_by_predicted"] != metrics["resnet18"]["confusion_actual_by_predicted"]
        or reference["confusion_actual_by_predicted"] != metrics["nearest_centroid"]["confusion_actual_by_predicted"]
    ):
        raise RuntimeError("The serving run differs from the recorded holdout evaluation.")
    return {"cohort": "confirmation" if new_per_class else "holdout", "rows": len(actual),
            "class_order": labels, "endpoint": endpoint, "nearest_centroid": reference,
            "first_handler_ms": latency[0], "warm_handler_median_ms": float(np.median(latency[1:]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--confirmation-per-class", type=int, default=0)
    args = parser.parse_args()
    if args.cache_dir:
        cfg.CACHE_DIR = args.cache_dir
    print(json.dumps(check_endpoint(args.run_dir, args.confirmation_per_class), indent=2))


if __name__ == "__main__":
    main()
