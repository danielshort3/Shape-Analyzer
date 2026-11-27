# Five-shape sketch classifier

## Intended use

This CPU benchmark evaluates a small configuration of a five-way ResNet18 classifier for isolated strokes on a dark square canvas. It demonstrates splitting, evaluation and local inference. A softmax score measures confidence within these five choices. The model has no calibrated probability guarantee or out-of-distribution rejection.

## Data, training and comparison

Google's Quick, Draw! bitmap release stores a 28 × 28 grayscale drawing per row in each category's file. The study used the circle, triangle, square, hexagon and octagon files. A deterministic per-class sample produced 510 training, 130 validation and 200 held-out drawings. Source file content and class-specific row indices determine membership. A separate confirmation sample of 200 excludes every previously used row. The bitmap files contain no artist identity for a grouped split. Visually duplicate drawings across different source IDs remain possible.

The one-channel ResNet18 starts with random weights. Training uses cross-entropy, AdamW, and random affine, blur, jitter, erasing and pixel-noise augmentation. The quick profile uses 64 × 64 nearest-neighbor input, batch size 32, seed 42, four CPU epochs and a learning rate of 0.0003. The checkpoint with lowest validation loss is exported. The reference predicts the class with the closest training-class mean image by mean squared pixel distance. Both methods use the same training and held-out source rows. No model choice or parameter was selected using held-out class labels.

## Measured results

| Cohort | Observations | ResNet18 accuracy | Centroid accuracy |
|---|---:|---:|---:|
| Reserved holdout | 200 | 124 / 200 (62.0%) | 131 / 200 (65.5%) |
| Independent confirmation | 200 | 122 / 200 (61.0%) | 118 / 200 (59.0%) |

The direction of the difference changes between these small cohorts. A larger independent study would be needed to rank the two methods reliably. Each cohort contains 40 drawings per class, so accuracy and macro recall coincide.

| Class | Holdout correct / 40 | Model recall | Centroid recall | Confirmation model recall |
|---|---:|---:|---:|---:|
| Circle | 6 | 15.0% | 67.5% | 30.0% |
| Triangle | 39 | 97.5% | 92.5% | 97.5% |
| Square | 37 | 92.5% | 67.5% | 90.0% |
| Hexagon | 14 | 35.0% | 67.5% | 20.0% |
| Octagon | 28 | 70.0% | 32.5% | 67.5% |

**ResNet18 holdout confusion matrix**. Rows are actual classes and columns are predictions.

| Actual class | Circle | Triangle | Square | Hexagon | Octagon |
|---|---:|---:|---:|---:|---:|
| Circle | 6 | 0 | 0 | 1 | 33 |
| Triangle | 0 | 39 | 0 | 1 | 0 |
| Square | 0 | 0 | 37 | 0 | 3 |
| Hexagon | 3 | 0 | 0 | 14 | 23 |
| Octagon | 5 | 1 | 1 | 5 | 28 |

Of 40 circles, 33 were labeled octagon. Of 40 hexagons, 23 were labeled octagon. Treat these as measured error patterns in this run. Differences in stroke thickness, rotational appearance or training budget would require a separately designed experiment.

## Train/serve parity and reproducibility

The historical serving path resized the short side to 256 and cropped a 224-pixel center region. Applied to the same frozen weights, that legacy path recognized 40 of the original 200 held-out drawings. The trainer's held-out evaluation recognized 124. The new sidecar addresses the input mismatch by reproducing the recorded canvas size, nearest-neighbor resize and normalization before inference. The model weights stayed fixed during this check. With the sidecar, the actual API handler reproduced the original confusion matrix. The subsequent confirmation test used previously unseen source rows.

Results were measured on a Windows x86-64 CPU using Python 3.12.11, PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu. Seeds and content digests allow the split to be reproduced. Library releases, hardware, thread scheduling and accelerators can affect learned weights. The test suite checks class-by-class source isolation and exact preprocessing tensor agreement. This study covers the quick configuration. Full-scale, GPU and hosted Lambda measurements remain separate.

The [README](../README.md) gives replay commands. The [full results](quickdraw-results.json) include class precision, both confusion matrices, source SHA-256 values and sample counts. Google, Inc. publishes the original drawings under CC BY 4.0; selecting rows, resizing images and applying training augmentation are the transformations used here. [Official dataset](https://github.com/googlecreativelab/quickdraw-dataset) · [Attribution terms](https://creativecommons.org/licenses/by/4.0/)
