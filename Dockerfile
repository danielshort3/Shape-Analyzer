FROM public.ecr.aws/lambda/python:3.12

COPY requirements-inference.txt .

RUN pip install --no-cache-dir -r requirements-inference.txt

COPY inference.py .

COPY shape_analyzer ./shape_analyzer/

# Model weights
COPY checkpoints/*.ckpt ./model.ckpt
ENV MODEL_PATH="model.ckpt"

CMD ["inference.handler"]
