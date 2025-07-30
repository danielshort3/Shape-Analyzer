# Shape-Analyzer

## Setup

Install the required Python packages before running the training or inference scripts:

```bash
pip install -r requirements.txt
```

Training uses the Lightning framework. After training completes, the script
automatically extracts the raw weights from the best `.ckpt` file and writes
`model.pt` into the `checkpoints` directory. The inference script loads this
`.pt` file directly without needing Lightning installed.
