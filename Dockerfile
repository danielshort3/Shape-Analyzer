# ── Base image optimised for AWS Lambda (Python 3.12) ───────────────────
FROM public.ecr.aws/lambda/python:3.12

# ----------------------------------------------------------------------
# 1. Install Python dependencies to /opt  (auto‑added to PYTHONPATH)
# ----------------------------------------------------------------------
# Put ONLY the libs your model really needs into requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t /opt

# ----------------------------------------------------------------------
# 2. Add inference code and model weights
# ----------------------------------------------------------------------
COPY inference.py   .
COPY model.pt       .           # rename / path‑adjust as needed

# ----------------------------------------------------------------------
# 3. Tell Lambda which function to run
# ----------------------------------------------------------------------
#  (file_name.function_name)
CMD ["inference.handler"]
