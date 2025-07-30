# Shape-Analyzer

## Setup

Install the required Python packages before running the inference script:

```bash
pip install -r requirements.txt
```

Training the model produces `model.pt` inside the `checkpoints` directory. The
inference script expects this `.pt` file rather than a Lightning checkpoint.
