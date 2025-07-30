# ── Base image optimised for AWS Lambda (Python 3.12) ───────────────────
FROM public.ecr.aws/lambda/python:3.12

# ----------------------------------------------------------------------
# 1. Install Python dependencies to /opt  (auto‑added to PYTHONPATH)
# ----------------------------------------------------------------------
# Put ONLY the libs your inference code needs into requirements-inference.txt
COPY requirements-inference.txt .
RUN pip install --no-cache-dir --no-compile -r requirements-inference.txt -t /opt

# ----------------------------------------------------------------------
# 2. Add inference code and model weights
# ----------------------------------------------------------------------
COPY inference.py   .
# Copy the only checkpoint file, renaming it to a stable path
COPY checkpoints/*.ckpt /tmp/
RUN mv /tmp/*.ckpt model.ckpt
ENV MODEL_PATH="model.ckpt"

# ----------------------------------------------------------------------
# 3. Tell Lambda which function to run
# ----------------------------------------------------------------------
#  (file_name.function_name)
CMD ["inference.handler"]
