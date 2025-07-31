# Shape-Analyzer

Shape‑Analyzer is a small demonstration project that recognises basic hand drawn
shapes.  It provides a simple end‑to‑end pipeline covering data preparation,
model training and an example inference endpoint.  The project is intentionally
minimal and is intended as a teaching and recruitment sample rather than a
production system.

The model predicts one of five shapes – circle, triangle, square, hexagon or
octagon – from a single monochrome image.  A compact ResNet18 encoder is trained
using PyTorch Lightning.  Training data comes from the publicly available Google
QuickDraw sketch dataset, with helper utilities to download and cache the
relevant classes.  The repo also contains optional code for generating synthetic
outlines to experiment with data augmentation.

## Setup

Install the required packages before running the training or inference scripts:

```bash
pip install -r requirements.txt
```

Training uses the Lightning framework.  After training completes, the script
automatically extracts the raw weights from the best `.ckpt` file and writes a
`model.pt` file into the `checkpoints` directory.  The inference script loads
this `.pt` directly without needing Lightning installed.

## Usage

**Train a model**

```bash
python shapes_pipeline.py
```

This downloads the QuickDraw shape sketches (a few hundred megabytes), builds
the training and validation splits and trains a classifier.  The final weights
are written to `checkpoints/model.pt`.

**Try the interactive demo**

```bash
python - <<'PY'
from pathlib import Path
from shape_analyzer.demo import interactive_demo
interactive_demo(Path('checkpoints/model.pt'))
PY
```

Use the mouse to draw on the canvas and press **s** to see the predicted class.

**Serverless inference**

`inference.py` implements a minimal AWS Lambda handler that expects a
base‑64‑encoded image in the request body.  It returns the predicted class and
confidence as JSON.  See `local_test.py` for an example of invoking the handler
locally.

## Tests

Unit tests cover the dataset utilities and help ensure the demo code runs on new
environments.  After installing the requirements you can run:

```bash
pytest
```

This repository is lightweight and self‑contained, making it a straightforward
example for demonstrating machine learning proficiency to recruiters.
