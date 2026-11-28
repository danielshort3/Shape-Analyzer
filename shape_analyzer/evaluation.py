"""Measure the selected model and a training-only nearest-centroid reference."""
import numpy as np
import torch


def classification_report(actual, predicted, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    correct = matrix.diagonal()
    support = matrix.sum(axis=1)
    chosen = matrix.sum(axis=0)
    recall = np.divide(correct, support, out=np.zeros(len(labels)), where=support > 0)
    precision = np.divide(correct, chosen, out=np.zeros(len(labels)), where=chosen > 0)
    return {
        "accuracy": float(correct.sum() / matrix.sum()),
        "macro_recall": float(recall[support > 0].mean()),
        "per_class": {
            name: {"support": int(support[i]), "correct": int(correct[i]),
                   "precision": float(precision[i]), "recall": float(recall[i])}
            for i, name in enumerate(labels)
        },
        "confusion_actual_by_predicted": matrix.tolist(),
    }


def evaluate(model, train_dataset, test_loader, labels):
    train_idx = train_dataset.idxs
    centroids = np.stack([
        train_dataset.imgs[train_idx[train_dataset.cls[train_idx] == cid]].mean(axis=0, dtype=np.float64)
        for cid in range(len(labels))
    ])
    test = test_loader.dataset
    # A batch at a time keeps the reference usable on the full-size holdout.
    reference = []
    for start in range(0, len(test), 512):
        pixels = test.imgs[test.idxs[start:start + 512]].astype(np.float64)
        distances = np.stack([np.square(pixels - mean).mean(axis=(1, 2)) for mean in centroids], axis=1)
        reference.extend(distances.argmin(axis=1))
    predicted, actual = [], []
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for x, y, _ in test_loader:
            predicted.extend(model(x.to(device)).argmax(dim=1).cpu().tolist())
            actual.extend(y.tolist())
    return {"class_order": list(labels), "holdout_drawings": len(actual),
            "resnet18": classification_report(actual, predicted, labels),
            "nearest_centroid": classification_report(actual, reference, labels)}
