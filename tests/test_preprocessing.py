import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

from shape_analyzer.preprocessing import (
    normalized_tensor, prediction_transform, preprocess_description, resize_bitmap,
)


def test_new_weight_contract_reproduces_training_tensor_and_checks_identity(tmp_path):
    # A stand-in byte sequence suffices here because this path only hashes the file.
    weight = tmp_path / "model.pt"
    weight.write_bytes(b"locally generated identity fixture")
    names = ["circle", "square"]
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    weight.with_suffix(".preprocess.json").write_text(
        json.dumps(preprocess_description(64, names, digest)), encoding="utf-8"
    )
    raw = Image.fromarray(np.arange(28 * 28, dtype=np.uint8).reshape(28, 28))
    result = prediction_transform(weight, names, None)(raw)
    assert torch.equal(result, normalized_tensor()(resize_bitmap(raw, 64)))
    with pytest.raises(ValueError, match="incompatible"):
        prediction_transform(weight, list(reversed(names)), None)
    weight.write_bytes(b"another local identity")
    with pytest.raises(ValueError, match="different model"):
        prediction_transform(weight, names, None)


def test_weights_without_metadata_keep_the_provided_legacy_transform(tmp_path):
    previous = object()
    assert prediction_transform(tmp_path / "old.pt", ["a"], previous) is previous
