# Shape Analyzer

A five-class image classifier for hand-drawn circles, triangles, squares, hexagons and octagons. The project trains a one-channel ResNet18, evaluates a validation-selected checkpoint against a nearest-centroid reference, and serves the exported weights through a local or Lambda-compatible JSON handler.

In one four-epoch CPU run, the model recognized **124 of 200** held-out Google Quick, Draw! sketches (62.0%). The centroid reference recognized **131** (65.5%). On another 200 previously unused sketches, the model recognized 122 (61.0%) and the reference recognized 118 (59.0%). Circles and hexagons were frequently confused with octagons. [Model card, per-class results and limitations](docs/model-card.md)

## Run a small offline example

Use **64-bit CPython 3.12**. The CPU dependency lock includes the test tools. On Windows create and activate a virtual environment with `py -3.12 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`. On Linux use `python3.12 -m venv .venv` and `. .venv/bin/activate`.

```sh
python -m pip install --require-hashes -r requirements-cpu.txt
python shapes_pipeline.py --demo --device cpu --no-progress
python local_test.py --model shape-run-demo/checkpoints/model.pt --draw-circle
python -m shape_analyzer.evaluate_serving --run-dir shape-run-demo
python -m pytest -q tests
```

After dependencies are installed, the demo draws its own small geometry fixture and runs one real training epoch, evaluation, checkpoint export and inference locally. The resulting class and softmax score illustrate how to use the model. The Google sketch results above come from a separate run.

## Reproduce the sketch benchmark

```sh
python shapes_pipeline.py --quick --device cpu --cache-dir qd_cache --no-progress
python -m shape_analyzer.evaluate_serving --run-dir shape-run-quick --cache-dir qd_cache
python -m shape_analyzer.evaluate_serving --run-dir shape-run-quick --cache-dir qd_cache --confirmation-per-class 40
```

The first run downloads five official bitmap collections (527,710,800 bytes in the measured snapshot). The same raw cache can be reused. Seed 42 selects **102 training, 26 validation and 40 held-out drawings per class**. The confirmation run draws another 40 per class from the complement of all three partitions. The class, source content digest and original row index identify each drawing. Changing input content, sampling settings or seed creates new split files. [Machine-readable benchmark results](docs/quickdraw-results.json)

The original full training command remains `python shapes_pipeline.py`. Its defaults request 50,000 development and 10,000 holdout samples per class with 256-pixel input. Use `--work-dir`, `--cache-dir`, `--seed`, `--device` and `--epochs` to control a run. Quick runs use a 64-pixel canvas. See `python shapes_pipeline.py --help` for invocation details. The reported study used CPU. A GPU run requires a driver-compatible PyTorch and torchvision build. [PyTorch installation options](https://pytorch.org/get-started/locally/)

## Architecture and outputs

The data builder makes one class-specific sample without replacement and partitions source rows before applying training-only augmentation. The validation loss selects the checkpoint. The holdout produces a confusion matrix, accuracy, macro recall, per-class precision and recall. A nearest-centroid reference averages only training images. [Split implementation](shape_analyzer/splits.py) · [Evaluation](shape_analyzer/evaluation.py)

Each run exports three files in its checkpoints directory.

| Output | Purpose |
|---|---|
| `model.pt` | Plain state dictionary for the original five-class ResNet18 architecture. |
| `model.preprocess.json` | Class order, trained canvas size, resizing policy and SHA-256 of the corresponding weights. Keep it beside the weights when serving a newly trained model. |
| `evaluation.json` | Run configuration, source content digests, selected validation loss, class metrics and both confusion matrices. |

The `build_qd_dataset()` and `build_qd_holdout()` helpers return their versioned archive paths. Legacy `shapes_data.npz` and `qd_holdout.npz` files lack source IDs and remain available for explicit reading with `NPShapes(path)`. Code that opens generated splits directly should use the path returned by the builder.

## Inference and drawing

Set `MODEL_PATH` to a trusted state dictionary. New exports use the training canvas and normalization. Existing weights without a sidecar use the historical resize-and-crop transform. The local command draws a fresh circle by default or accepts your image through `--image PATH`. To install a separate CPU serving environment, run `python -m pip install -r requirements-inference.txt`.

| Route | Behavior |
| --- | --- |
| `GET /` or `GET /health` | Returns the service name, model-loaded flag, label order and request limit. A fresh process serves health using only the Python standard library. |
| `GET /warmup` or `POST /warmup` | Loads the configured model, runs a blank grayscale input through it and returns all five scores and elapsed handler milliseconds. |
| `POST /`, `POST /predict` or `POST /classify` | Accepts a JSON object with a `b64` image string. Returns `class`, `confidence`, per-label `shape_confidences` and `duration_ms`. |

Predictions also accept direct Lambda events containing the JSON body without HTTP routing metadata, plus API Gateway events with a base64-encoded binary image body. The encoded image is limited to 500,000 characters and 16,777,216 decoded pixels. The reported scores are rounded softmax probabilities across the five training classes.

For a mouse-driven desktop canvas, install the optional `requirements-demo.txt` in your training environment. Run `python -c "from pathlib import Path; from shapes_pipeline import interactive_demo; interactive_demo(Path('shape-run-quick/checkpoints/model.pt'))"`. Use **s** to predict, **r** to reset and **q** to quit. The CLI provides a keyboard-accessible alternative.

The Lambda Dockerfile reads the state dictionary and optional preprocessing sidecar from the repository's `checkpoints/` directory. The build context restricts inputs to model, inference code and its dependency manifest. The local handler was exercised with actual image payloads. Container and hosted AWS verification are separate release checks.

## Project history and data rights

The original project supplied the ResNet18 training pipeline, Quick Draw and synthetic image utilities, desktop drawing interface and Lambda handler. Modernization added verifiable source-row partitions, cache migration, comparative evaluation, a train/serve preprocessing contract and an offline smoke run.

Daniel Short licenses his original Python source, tests, build files and CI configuration under MIT. Dependencies retain their authors' licenses. Source licensing does not grant rights to datasets, image assets or model weights. [MIT license](LICENSE)

Google, Inc. makes the Quick, Draw! dataset available under CC BY 4.0. The benchmark selects and resizes bitmaps for training and evaluation. The project downloads the raw drawings and produces model weights locally. The offline example draws its own geometry. [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
