# IFRPM -- Intelligent Fleet Risk and Predictive Maintenance

IFRPM is a system built to predict when aircraft components are likely to fail, so maintenance crews can act before problems happen rather than after. It combines machine learning models trained on real flight and sensor data with a FastAPI backend and a React dashboard, giving fleet operators a single place to monitor aircraft health, track remaining useful life (RUL), and respond to risk alerts.

The project is split into two main phases. Phase 1 handles data ingestion, preprocessing, and model training. Phase 2 wraps those trained models into a production-ready API with a web-based monitoring dashboard.


## How It Works

The system pulls in data from multiple sources -- NGAFID flight records, NASA C-MAPSS turbofan degradation datasets, battery cycling data, and electrical component measurements. These go through a cleaning pipeline that removes duplicates, fills gaps, strips outliers, and produces normalized datasets ready for training.

Two model architectures are available for training: a standard CNN and a convolutional multi-head self-attention model (Conv-MHSA). Both are configured through a single YAML file and output trained weights that the backend can load at startup.

On the backend side, a FastAPI server picks up any `.pkl` or `.h5` model files from the models directory and runs inference across all of them. RUL predictions are averaged across loaded models, so you can swap models in and out without code changes. If a model file is missing, the server skips it gracefully instead of crashing.

The React dashboard connects to the API and displays fleet-wide health summaries, per-aircraft component status, historical RUL trends, risk bands, and weather-correlated alerts.


## Project Layout

```
IFRPM/
├── backend/                 FastAPI application
│   └── app/
│       ├── main.py          Application entry point
│       ├── config.py         Settings (DB, thresholds, model dir)
│       ├── database.py       SQLAlchemy session management
│       ├── seed.py           Initial data seeding
│       ├── ml/               Model loading and inference
│       ├── models/           Database models (aircraft, components, predictions)
│       ├── routers/          API endpoints (fleet, aircraft, RUL, alerts, weather)
│       ├── schemas/          Pydantic request/response schemas
│       ├── services/         Business logic (risk scoring, RUL calculation, weather)
│       └── utils/            Feature engineering and health index math
├── phase_1/                 Data pipeline and model training
│   ├── download.py          Dataset downloader and converter
│   ├── preprocess.py        Cleaning pipeline (dedup, fill, outlier removal)
│   ├── main.py              Training entry point
│   ├── configs/             YAML configuration
│   ├── data/                Raw and processed datasets
│   ├── models/              Saved model weights
│   ├── tasks/               Training and explainability tasks
│   ├── preprocessing/       Dataset-specific loaders
│   ├── scripts/             Helper scripts
│   └── utils/               Logging, seeding, general utilities
├── src/
│   └── dashboard/           React frontend (Vite + React 19)
├── docs/
│   ├── api.md               Full API reference
│   └── backend.md           Backend architecture notes
└── requirements.txt         Python dependencies
```


## Getting Started

### Prerequisites

- Python 3.10 or later
- Node.js 18+ and npm
- PostgreSQL (the backend defaults to `localhost:5432` with database `ifrpm_dev`)
- A CUDA-capable GPU is recommended for training but not required

### Setting Up the Backend

Install Python dependencies from the project root:

```
pip install -r requirements.txt
```

Create a `.env` file in `backend/` based on `.env.example`, or rely on the defaults (PostgreSQL at localhost, user `ifrpm`, password `ifrpm`, database `ifrpm_dev`).

Start the API server:

```
cd backend
uvicorn app.main:app --reload
```

The server will initialize the database, load any model files from the `models/` directory, and seed sample data on first run. API docs are available at `http://localhost:8000/docs`.

### Running the Dashboard

```
cd src/dashboard
npm install
npm run dev
```

The dashboard runs on `http://localhost:5173` by default and expects the API at `http://localhost:8000`.

### Training Models (Phase 1)

First, download and process the datasets:

```
cd phase_1
pip install -r requirements.txt
python download.py
python preprocess.py
```

Place source ZIP files in `phase_1/data/` before running `download.py`. The script will extract, convert, and validate data from all supported sources into `data/processed/`.

To train a model:

```
python main.py --model conv_mhsa --size_ratio 0.5
```

Options:
- `--model`: `cnn` or `conv_mhsa`
- `--size_ratio`: fraction of the dataset to use (useful for quick iteration)
- `--task`: `train`, `explain`, or `all`
- `--debug true`: enables debug mode with verbose logging
- `--config`: path to a custom YAML config file

Trained weights land in `outputs/models/` and can be copied to the backend `models/` directory for serving.


## Supported Datasets

- **NGAFID** -- General aviation flight data from the National General Aviation Flight Information Database. Loaded from Parquet files.
- **NASA C-MAPSS** -- Turbofan engine degradation simulation data. Used for RUL regression benchmarking.
- **Battery cycling** -- Charge/discharge cycle data for lithium-ion battery degradation modeling.
- **Electrical components** -- Capacitor and resistor degradation measurements.

All datasets are converted to a common format during preprocessing, output as both CSV and PKL files.


## API Overview

The backend exposes five route groups under `/api/v1`:

- `/fleet/summary` -- Fleet-wide health overview with per-aircraft risk bands
- `/fleet/{id}/history` -- Historical RUL predictions for a specific aircraft
- `/aircraft` -- Individual aircraft details and component listings
- `/rul` -- On-demand RUL predictions using loaded models
- `/alerts` -- Active maintenance alerts based on configurable thresholds
- `/weather` -- Weather data integration for operational risk assessment

Health check is at `GET /health`. Full endpoint documentation is in `docs/api.md`.


## Model Inference

The backend scans the `models/` directory at startup and loads every supported file it finds:

- Scikit-learn pickles (`.pkl`) are loaded via joblib
- Keras/TensorFlow weights (`.h5`) are compiled with TensorFlow

RUL endpoints return the averaged prediction across all loaded models. If you only have one model file, that single prediction is returned as-is. Missing files are skipped without errors, so the server works fine during development even with an incomplete model set.

Risk bands are assigned based on configurable thresholds in `config.py`:
- Critical: RUL below 10 cycles
- High: below 30
- Medium: below 80
- Low: 80 and above


## Configuration

Backend settings can be overridden with environment variables or a `.env` file:

- `DATABASE_URL` -- PostgreSQL connection string
- `MODEL_DIR` -- Path to the directory containing trained model files
- `RUL_CRITICAL_THRESHOLD`, `RUL_HIGH_THRESHOLD`, `RUL_MEDIUM_THRESHOLD` -- Cycle thresholds for risk classification

Training configuration lives in `phase_1/configs/config.yaml` and covers dataset paths, model architecture parameters, training hyperparameters, and output directories.


## License

See the repository for license details.
