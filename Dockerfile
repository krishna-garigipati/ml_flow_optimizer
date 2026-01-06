FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN mkdir -p /app/outputs

ENV PYTHONPATH=/app/src
ENV MLFLOW_TRACKING_URI=file:///app/outputs/mlruns

CMD ["python", "src/pipeline.py"]
