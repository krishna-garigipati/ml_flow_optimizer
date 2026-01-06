FROM python:3.9-slim

WORKDIR /app

# System dependencies (XGBoost + Plotly image export)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code INCLUDING dataset (src/data)
COPY src/ ./src/

# Create outputs directory
RUN mkdir -p /app/outputs

# Environment variables
ENV PYTHONPATH=/app/src
ENV MLFLOW_TRACKING_URI=file:///app/outputs/mlruns

# Run pipeline
CMD ["python", "src/pipeline.py"]
