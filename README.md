
# Lake FastF1

[![tests](https://github.com/rvanguita/lake-fastf1/actions/workflows/tests.yml/badge.svg)](https://github.com/rvanguita/lake-fastf1/actions/workflows/tests.yml)

![Lake FastF1](img/high.png)

Lake FastF1 is a long-term data engineering and machine learning project focused on Formula 1 data. It combines data ingestion, lakehouse-style transformations, and a lightweight prediction experience powered by a trained model.

## What this project does

The project collects Formula 1 race and session results using the FastF1 library, processes them through a multi-layer data pipeline, and prepares curated datasets for analytics and prediction.

The workflow includes:

- Extracting raw race/session results from Formula 1 events.
- Storing the raw data in Parquet files.
- Consolidating the data into a Bronze layer with Delta Lake.
- Creating Silver-layer datasets for champions, driver statistics, and analytical tables.
- Training a championship-prediction model (scikit-learn) tracked in MLflow.
- Mirroring the Bronze/Silver Delta tables into MySQL for BI consumers outside the Spark/Delta stack.
- Exposing predictions through a **FastAPI** service and an interactive **Streamlit** dashboard.
- Optionally archiving raw Parquet files to S3 (manual utility, not part of the DAG).

## Architecture

The project follows a practical data platform structure:

- **Raw layer**: initial extracted results stored as Parquet files.
- **Bronze layer**: consolidated data tables stored in Delta format.
- **Silver layer**: curated analytical datasets prepared for reporting and model consumption.
- **MySQL mirror**: the last DAG step (`src/sender_local.py`) copies every Bronze/Silver Delta table into MySQL for consumers outside the Spark/Delta stack.
- **Orchestration**: the Airflow DAG `data-pipeline` (`dags/data_pipeline.py`) runs weekly on Mondays (`catchup=False`, `max_active_runs=1`), wiring `raw >> bronze >> silver >> sender_mysql` with explicit `Asset` lineage.
- **S3 archive**: `src/sender.py` is a standalone CLI (not wired into the DAG) that uploads raw Parquet files to S3.
- **Serving layer**: a FastAPI service and a Streamlit dashboard consume the processed data and machine learning outputs.

```
┌────────────────────────── Airflow DAG: data-pipeline (weekly) ──────────────────────────┐
│                                                                                         │
│  FastF1 ──► Raw (Parquet) ──► Bronze (Delta) ──► Silver (Delta) ──► MySQL mirror         │
│                 │                                                                        │
└─────────────────┼───────────────────────────────────────────────────────────────────────┘
                  │ manual CLI: src/sender.py
            ┌─────▼─────┐
            │ S3 bucket │
            └───────────┘

   MLflow (tracking) ──model──► FastAPI :5002 /predict ◄──── Streamlit :8501 dashboard
                                                              │
                        Bronze + Silver (Delta) ──read────────┘
```

## Main technologies

| Layer | Technology |
|---|---|
| Data access | FastF1 |
| Data processing | Pandas, NumPy, PySpark, `deltalake` (direct Delta reads in the API/dashboard) |
| Storage | Delta Lake (Bronze/Silver), Parquet (Raw) |
| BI mirror | MySQL (PyMySQL + SQLAlchemy) |
| Object storage | AWS S3 (boto3) |
| Orchestration | Apache Airflow |
| Modeling | scikit-learn |
| Model tracking | MLflow |
| Prediction API | **FastAPI** + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Package management | **uv** |
| Deployment | Docker, Docker Compose |

## Repository structure

```
lake-fastf1/
├── app/
│   ├── api/                    # FastAPI prediction service (own uv project)
│   │   ├── main.py
│   │   ├── pyproject.toml / uv.lock
│   │   └── Dockerfile
│   └── streamlit/              # Interactive dashboard (own uv project)
│       ├── main.py
│       ├── pyproject.toml / uv.lock
│       └── Dockerfile
├── dags/
│   └── data_pipeline.py        # the single Airflow DAG (dag_id="data-pipeline")
├── src/                        # ETL / ML source
│   ├── extract_data.py         # Raw: FastF1 -> data/raw (Parquet)
│   ├── spark_session.py        # Bronze: consolidate Raw -> Delta (+ Spark/Delta helpers)
│   ├── silver_data.py          # Silver: champions, driver_statistic_*, tb_abt
│   ├── train_driver_champion.py  # train model, log/register in MLflow
│   ├── sender_local.py         # mirror Bronze/Silver Delta tables into MySQL
│   ├── sender.py               # upload raw Parquet to S3 (standalone CLI)
│   ├── queries/                # SQL run by the Silver layer (champions.sql, ...)
│   └── lake_fastf1/            # packaged entry point (uv_build)
├── data/                       # Raw, Bronze, Silver datasets (gitignored)
├── requirements.txt            # deps for the Airflow image (installed via pip)
├── pyproject.toml              # root uv project
├── CLAUDE.md                   # architecture & development guide
├── .env.example
├── docker-compose.yml
└── Dockerfile                  # Airflow image
```

## Getting started

Run the project locally with Docker Compose:

```bash
docker compose up --build -d
```

Services exposed:

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 |
| FastAPI | http://localhost:5002 |
| FastAPI Swagger | http://localhost:5002/docs |
| Streamlit | http://localhost:8501 |

### Tests

`pytest`, split into three suites (one per `uv` project). They mock every external dependency
(FastF1, Spark, MLflow, MySQL, S3), so they run in seconds:

```bash
uv run pytest                        # root — src/ helpers + ExtractData
(cd app/api && uv run pytest)        # FastAPI routes
(cd app/streamlit && uv run pytest)  # dashboard pandas helpers
```

CI (`.github/workflows/tests.yml`) runs all three suites plus `ruff format --check` on every
push and pull request.

### Environment variables

Copy `.env.example` to `.env` and fill in the values. Every pipeline module (`src/*.py`) reads
its config via `os.environ[...]` and raises `KeyError` if a required var is unset — always run
through `docker compose` or with `.env` sourced.

```env
# AWS / S3 (raw Parquet archival: src/sender.py, manual CLI)
AWS_KEY=
AWS_SECRET_KEY=
REGION_NAME="us-east-1"

# Airflow (docker compose)
AIRFLOW_VERSION=3.3.0
AIRFLOW_PORT=8080
AIRFLOW_UID=1000

# FastAPI prediction service
API_PORT=5002

# Data lake layer paths + SQL directory
PATH_RAW="data/raw"
PATH_BRONZE="data/bronze"
PATH_SILVER="data/silver"
PATH_QUERIES="src/queries"

FORMAT_READ="parquet"

# MLflow
MLFLOW_URI="http://localhost:5050"
MLFLOW_MODEL_REGISTERED="f1-champion"
MLFLOW_EXPERIMENT_NAME="f1-champion"

# Streamlit dashboard
STREAMLIT_PORT=8501

# MySQL mirror (Bronze/Silver -> MySQL: src/sender_local.py, DAG task sender_mysql)
MYSQL_HOST=
MYSQL_PORT=3306
MYSQL_ID_TABLE=fastf1
MYSQL_USER=
MYSQL_PASSWORD=
```

| Variable | Description |
|---|---|
| `PATH_RAW` | Directory where extracted Parquet files are stored. |
| `PATH_BRONZE` | Directory where consolidated Bronze tables are stored. |
| `PATH_SILVER` | Directory where processed Silver tables are stored. |
| `PATH_QUERIES` | Directory containing the SQL files run by the Silver layer. |
| `FORMAT_READ` | File format read from the Raw layer (`parquet`). |
| `MLFLOW_URI` | Tracking server URI used by the API and training scripts. |
| `MLFLOW_MODEL_REGISTERED` | Name of the registered model served by the API. |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name used when logging training runs. |
| `API_PORT` | Port exposed by the FastAPI service (default `5002`). |
| `STREAMLIT_PORT` | Port exposed by the Streamlit dashboard (default `8501`). |
| `AIRFLOW_PORT` / `AIRFLOW_UID` / `AIRFLOW_VERSION` | Airflow container port, host UID for file permissions, and image version. |
| `AWS_KEY` / `AWS_SECRET_KEY` / `REGION_NAME` | Credentials/region for the manual S3 upload of raw Parquet (`src/sender.py`). |
| `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USER` / `MYSQL_PASSWORD` | Connection for the MySQL mirror step (`src/sender_local.py`). |
| `MYSQL_ID_TABLE` | Table-name prefix for the mirrored tables (`{MYSQL_ID_TABLE}_{layer}_{table}`). |

> The Streamlit container additionally gets `TABLE_PATH_SILVER` and `TABLE_PATH_BRONZE` from
> `docker-compose.yml` (paths of the read-only Delta mounts inside that container, not the same
> as `PATH_SILVER` / `PATH_BRONZE`), and reaches the API at `http://api-driver-champion:${API_PORT}`
> (the Docker Compose service name), not `localhost`.

## Dashboard

The Streamlit dashboard (`http://localhost:8501`) provides a full view of the F1 season — from raw race results to ML-based championship predictions.

### Code

`app/streamlit/` is four modules: `main.py` (page config, sidebar, layout),
`data.py` (Delta reads, API calls, `st.cache_data` wrappers), `analytics.py`
(pure pandas aggregates + `build_insights`), `charts.py` (theme-aware Plotly
builders). Charts follow the viewer's light/dark Streamlit theme.

### Layout

**Sidebar:** season selector (single), driver multi-select (defaults to top-5 by
probability, shown as `M. Verstappen — Red Bull Racing`), and a momentum /
recent-race window slider (3–10).

**Top of page — always visible:** championship strip (rounds · points leader ·
gap to P2 · most wins), top-5 probability cards (headshot · win % · Δ vs last
round), and an auto **Insights** panel — up to five deterministic findings
(biggest probability mover, fastest-rising form, closest teammate fight, best
qualifier, reliability watch).

**Tabs:**

| Tab | Content |
|---|---|
| 🔮 Prediction | Win probability over time (selected drivers, team colors, unified hover); **Momentum** — Δ-since-last-round bars + rank-change table; **Explainability** expander — global RF feature importance (from `GET /model_info`) plus a per-driver "top factors" heuristic (importance × above/below the field median) |
| 📅 Season | Season snapshot KPIs, points ranking bar, cumulative points progression, recent-race cards (grid → finish ▲/▼), finishing-position-by-round heatmap |
| 🔬 Deep Dives | Sub-tabs — Drivers (full season table with headshots), Constructors (points bar + table), **Teammates** (intra-constructor race/quali/points head-to-head), **Quali & reliability** (DNF rate, points-finish rate, grid stats + grid → finish spread box plot) |
| 📋 Data | Win probability pivot + full feature table (nulls preserved) |

## API

The FastAPI service serves the latest registered MLflow model, cached for
`MODEL_CACHE_TTL` seconds (default 300) so most requests skip the registry
round-trip.

```bash
# Health check
curl http://localhost:5002/health_check

# Model features + random-forest importances
curl http://localhost:5002/model_info

# Predict
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"values": [{"id": "...", "feature_1": 1.0, ...}]}'
```

Interactive documentation is available at `http://localhost:5002/docs`.

## Why this is a long-term project

This repository is designed as a long-term initiative rather than a one-off experiment. The pipeline is expected to evolve continuously as new data sources, validation rules, performance improvements, and modeling requirements are introduced.

Ongoing areas of improvement:

- Data quality checks and pipeline reliability
- Scheduling and backfill behavior
- Table design and business logic
- Model retraining and evaluation
- Observability, monitoring, and operational robustness
