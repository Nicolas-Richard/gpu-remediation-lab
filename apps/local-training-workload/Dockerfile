# Local CPU-only distributed recovery fixture.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY train.py /app/train.py

CMD ["python", "/app/train.py"]
