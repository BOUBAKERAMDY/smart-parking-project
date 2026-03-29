# Smart Parking Predictor

An end-to-end MLOps pipeline that predicts parking occupancy in Melbourne using a hybrid LSTM + GRU deep learning model, served via a FastAPI inference API and containerized with Docker.

## Architecture

```
Raw CSV Data --> ETL Pipeline --> Hourly Time Series --> LSTM+GRU Model --> FastAPI API
```

1. **ETL**: Raw sensor events are validated, normalized to UTC, binary-encoded, and resampled into hourly occupancy rates with full logging and error handling
2. **Model**: A hybrid LSTM (long-term) + GRU (short-term) architecture with a 12-hour lookback window
3. **API**: FastAPI serves predictions with strict Pydantic validation and operational metrics tracking
4. **Docker**: Slim production container with only inference dependencies
5. **CI/CD**: GitHub Actions pipeline runs linting and tests on every push

## Repository Structure

```
smart-parking-project/
├── .github/workflows/ci.yml  # CI/CD pipeline (lint + test)
├── data/                      # Raw and processed CSVs
├── models/                    # Binary artifacts (.keras, .pkl) - Git ignored
├── tests/                     # pytest test suite
│   ├── test_train.py          # ETL and sequence creation tests
│   └── test_app.py            # API endpoint tests
├── train.py                   # ETL Pipeline + Model Training script
├── app.py                     # FastAPI Application (Inference + Metrics)
├── config.yaml                # Centralized pipeline configuration
├── Dockerfile                 # Production container definition
├── Makefile                   # Pipeline orchestration commands
├── requirements.txt           # Frozen dependencies
└── README.md                  # Documentation
```

## Setup

**Prerequisites:** Python 3.9+

```bash
pip install -r requirements.txt
```

Or using Make:

```bash
make install
```

## Configuration

All pipeline parameters are centralized in `config.yaml`:

- **Data paths** and column mappings
- **Model hyperparameters** (lookback window, layer sizes, epochs, dropout)
- **API settings** (host, port, version)

## Training

Run the training pipeline to process raw data and train the model:

```bash
make train
```

This produces:
- `models/parking_model.keras` -- Trained Keras model
- `models/scaler.pkl` -- Fitted MinMaxScaler
- `data/processed_occupancy.csv` -- Processed hourly time series

The pipeline includes:
- **Data validation** -- Schema checks, value range validation, null detection
- **Structured logging** -- Every ETL step is logged with timestamps
- **Error handling** -- Graceful failure with informative error messages

## API Usage

Start the development server:

```bash
make serve
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict next-hour occupancy |
| GET | `/metrics` | Operational metrics (request count, latency, model version) |
| GET | `/docs` | Interactive Swagger documentation |

**Predict Example:**

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": [0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.6, 0.5, 0.4, 0.3, 0.5, 0.6]}'
```

**Metrics Example:**

```bash
curl http://localhost:8080/metrics
```

## Testing

Run the full test suite:

```bash
make test
```

Tests cover:
- **ETL pipeline** -- Output shape, value ranges, null handling, datetime index, hourly frequency
- **API endpoints** -- Response codes, schema validation, input rejection
- **Configuration** -- Config loading and required keys

## Linting

```bash
make lint
```

## Docker

Build and run the production container:

```bash
make docker-build
make docker-run
```

Or manually:

```bash
docker build -t smart-parking-api .
docker run -p 8080:8080 smart-parking-api
```

## CI/CD

GitHub Actions runs automatically on every push to `main`:
1. **Lint** -- Ruff checks code style and common errors
2. **Train** -- Runs the full ETL + training pipeline
3. **Test** -- Runs the pytest suite against trained artifacts

## Data Notes

- **Source**: Melbourne Open Data -- on-street parking bay sensors
- **Status Mapping**: Raw data uses `Present`/`Unoccupied`; `Present` is treated as occupied (1), `Unoccupied` as empty (0)
- **Timezone**: Raw timestamps have CET/CEST offsets (+01:00/+02:00), all normalized to UTC during ETL
- **Resampling**: Events are aggregated into hourly mean occupancy rates; gaps are forward-filled
