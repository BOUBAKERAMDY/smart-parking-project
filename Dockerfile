FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, config, model artifacts, and processed data
COPY app.py .
COPY config.yaml .
COPY models/ models/
COPY data/processed_occupancy.csv data/

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
