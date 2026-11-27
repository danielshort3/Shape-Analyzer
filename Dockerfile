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

# Package only the state dict exported from the selected validation checkpoint.
COPY checkpoints/ ./model/
RUN test -s ./model/model.pt
ENV MODEL_PATH="model/model.pt"

# ----------------------------------------------------------------------
# 3. Tell Lambda which function to run
# ----------------------------------------------------------------------
#  (file_name.function_name)
CMD ["inference.handler"]
