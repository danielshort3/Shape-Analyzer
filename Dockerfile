# ── Base image optimised for AWS Lambda (Python 3.12) ───────────────────
FROM public.ecr.aws/lambda/python:3.12

# ----------------------------------------------------------------------
# 1. Install Python dependencies to /opt  (auto‑added to PYTHONPATH)
# ----------------------------------------------------------------------
# Put ONLY the libs your inference code needs into requirements-inference.txt
COPY requirements-inference.txt .

RUN pip install --no-cache-dir --no-compile \
      -r requirements-inference.txt \
      --target /opt/python

# ----------------------------------------------------------------------
# 2. Add inference code and model weights
# ----------------------------------------------------------------------
COPY inference.py   .
COPY shape_analyzer /opt/python/shape_analyzer

# Copy the only checkpoint file, renaming it to a stable path
COPY checkpoints/*.pt /tmp/
RUN mv /tmp/*.pt model.pt
ENV MODEL_PATH="model.pt"

# ----------------------------------------------------------------------
# 3. Tell Lambda which function to run
# ----------------------------------------------------------------------
#  (file_name.function_name)
CMD ["inference.handler"]
