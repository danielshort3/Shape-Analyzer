import hashlib
import json
import pickle

import numpy as np
import pytest
import torch

from shape_analyzer.config import cfg
from shape_analyzer.data import make_loaders, train_tfm
from shape_analyzer.evaluate_serving import check_endpoint
from shape_analyzer.evaluation import classification_report, evaluate
from shape_analyzer.fixture import geometry_sources
from shape_analyzer.helpers import load_npz
from shape_analyzer.model import ShapeNet
from shape_analyzer.preprocessing import preprocess_description
from shape_analyzer.splits import prepare_splits


@pytest.fixture
def sources(monkeypatch, tmp_path):
    for name, value in {
        "N_PER_SHAPE": 10, "HOLD_QD_PER_SHAPE": 4, "TRAIN_FRAC": 0.7, "RNG_SEED": 42,
        "DATA_PATH": tmp_path / "shapes_data.npz", "HOLD_PATH": tmp_path / "qd_holdout.npz",
        "OUT_RES": 48, "BATCH_SIZE": 4, "WORKERS": 0,
    }.items():
        monkeypatch.setattr(cfg, name, value)
    return geometry_sources(18, 5)


def test_three_source_partitions_are_disjoint_for_every_class(sources):
    files = prepare_splits(sources, source_name="fixture")
    train, valid, test = make_loaders(files)
    for cid in range(len(cfg.CLASSES)):
        groups = []
        for loader in (train, valid, test):
            ds = loader.dataset
            indices = ds.idxs[ds.cls[ds.idxs] == cid]
            groups.append(set(ds.source_ids[indices].tolist()))
        assert [len(group) for group in groups] == [7, 3, 4]
        assert len(set.union(*groups)) == 14
    x, y, _ = next(iter(valid))
    assert x.shape == (4, 1, 48, 48) and y.dtype == torch.int64
    assert pickle.dumps(train_tfm)


def test_legacy_archives_are_preserved_and_never_used_for_evaluation(sources):
    for filename in (cfg.DATA_PATH, cfg.HOLD_PATH):
        np.savez(filename, imgs=np.zeros((1, 2, 2)), cls=[0], reg=[0])
    original = [p.read_bytes() for p in (cfg.DATA_PATH, cfg.HOLD_PATH)]
    files = prepare_splits(sources)
    assert files.development != cfg.DATA_PATH and files.holdout != cfg.HOLD_PATH
    assert [p.read_bytes() for p in (cfg.DATA_PATH, cfg.HOLD_PATH)] == original
    assert len(make_loaders(files)[2].dataset) == 20


def test_cache_identity_covers_contents_seed_and_counts(sources, monkeypatch):
    first = prepare_splits(sources)
    assert prepare_splits(sources) == first
    monkeypatch.setattr(cfg, "RNG_SEED", 43)
    reseeded = prepare_splits(sources)
    assert reseeded.development != first.development
    monkeypatch.setattr(cfg, "N_PER_SHAPE", 9)
    fewer = prepare_splits(sources)
    assert fewer.holdout != reseeded.holdout
    sources[cfg.CLASSES[0]][0, 0] ^= 255
    assert prepare_splits(sources).development != fewer.development


def test_invalid_cached_source_ids_are_regenerated(sources):
    files = prepare_splits(sources)
    before = load_npz(files.holdout)
    original = before["source_id"].copy()
    before["source_id"][:] = -1
    np.savez(files.holdout, **before)
    assert np.all(load_npz(files.holdout)["source_id"] == -1)
    rebuilt = prepare_splits(sources)
    assert np.array_equal(load_npz(rebuilt.holdout)["source_id"], original)


def test_insufficient_or_malformed_sources_fail_clearly(sources, monkeypatch):
    monkeypatch.setattr(cfg, "HOLD_QD_PER_SHAPE", 9)
    with pytest.raises(ValueError, match="distinct source rows"):
        prepare_splits(sources)
    sources[cfg.CLASSES[0]] = np.zeros((30, 17), dtype=np.uint8)
    with pytest.raises(ValueError, match="bitmap matrix"):
        prepare_splits(sources)


def test_confusion_rows_are_actual_and_columns_are_predictions():
    metrics = classification_report([0, 0, 1, 1], [0, 1, 1, 1], ("a", "b"))
    assert metrics["confusion_actual_by_predicted"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == 0.75 and metrics["macro_recall"] == 0.75
    assert metrics["per_class"]["b"]["precision"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("confirmation_count", [0, 1], ids=["holdout", "confirmation"])
def test_serving_rejects_another_checkpoints_matching_sidecar(tmp_path, confirmation_count):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    weights = checkpoints / "model.pt"
    weights.write_bytes(b"another run's weights")
    foreign_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    weights.with_suffix(".preprocess.json").write_text(
        json.dumps(preprocess_description(64, cfg.CLASSES, foreign_hash)), encoding="utf-8"
    )
    original_hash = hashlib.sha256(b"original weights").hexdigest()
    (checkpoints / "evaluation.json").write_text(json.dumps({"model_sha256": original_hash}), encoding="utf-8")

    # Check the training record before opening any cohort or deserializing weights.
    with pytest.raises(ValueError, match="checkpoint.*training"):
        check_endpoint(tmp_path, new_per_class=confirmation_count)


def test_serving_report_includes_digest_of_evaluated_geometry_model(tmp_path, sources):
    files = prepare_splits(sources, source_name="geometry fixture")
    train, _, test = make_loaders(files)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        model = ShapeNet()
    recorded = evaluate(model, train.dataset, test, cfg.CLASSES)
    checkpoint = tmp_path / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir()
    torch.save(model.state_dict(), checkpoint)
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    checkpoint.with_suffix(".preprocess.json").write_text(
        json.dumps(preprocess_description(cfg.OUT_RES, cfg.CLASSES, digest)), encoding="utf-8"
    )
    recorded.update(split=files.spec, model_sha256=digest)
    (checkpoint.parent / "evaluation.json").write_text(json.dumps(recorded), encoding="utf-8")

    report = check_endpoint(tmp_path)
    assert report["model_sha256"] == digest
    assert report["endpoint"] == recorded["resnet18"]
    assert report["nearest_centroid"] == recorded["nearest_centroid"]


def test_manual_archive_pair_with_duplicate_source_ids_is_rejected(sources):
    files = prepare_splits(sources)
    dev = load_npz(files.development)
    hold = load_npz(files.holdout)
    hold["source_id"][0] = dev["source_id"][0]
    np.savez(files.holdout, **hold)
    with pytest.raises(ValueError, match="more than once"):
        make_loaders(files)
