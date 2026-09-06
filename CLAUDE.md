# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A long-term data engineering / ML project for Formula 1 data. It pulls race and session results with the FastF1 library, runs them through a Raw → Bronze → Silver medallion pipeline (Parquet → Delta Lake, transformed with PySpark/SQL), trains a championship-prediction model tracked in MLflow, and serves predictions through a FastAPI service consumed by a Streamlit dashboard. Orchestration is Apache Airflow.

## Commands

Root project uses `uv` (Python >=3.13). `app/api` and `app/streamlit` are separate `uv` projects with their own `pyproject.toml`.

```bash
# Install root deps
uv sync

# Lint (ruff, no repo-specific config — uses defaults)
uv run ruff check .
uv run ruff format .

# Run the whole stack (Airflow :8080, FastAPI :5002, Streamlit :8501)
docker compose up --build -d

# Run pipeline stages locally/ad hoc (each module has a __main__)
uv run python -m src.extract_data      # FastF1 -> data/raw (Parquet)
uv run python -m src.spark_session     # Bronze: consolidate Raw parquet -> Delta
uv run python -m src.silver_data       # Silver: champions, driver_statistic_*, driver_all_statistic, tb_abt
uv run python -m src.train_driver_champion  # train model, log to MLflow

# Airflow (DAG lives in dags/data_pipeline.py, dag_id="data-pipeline")
# When run via docker compose, AIRFLOW_HOME is /opt/airflow/project/.airflow
# and the DAGs folder is /opt/airflow/project/dags (repo root mounted at /opt/airflow/project)

# Tests (pytest). Three separate suites — each its own uv project:
uv run pytest                              # root: src/ (tests/)
uv run pytest --cov=src --cov-report=term-missing
(cd app/api && uv run pytest)              # FastAPI routes (app/api/tests/)
(cd app/streamlit && uv run pytest)        # dashboard pandas helpers (app/streamlit/tests/)
```

The test suites are deliberately infra-free: FastF1 (network), Spark/Delta (JVM), MLflow, MySQL
and S3 are all mocked or replaced with `tmp_path`, so `pytest` runs in seconds. What's covered:

- **root (`tests/`)** — `ExtractData` (FastF1 mocked), `sender_local.find_delta_tables` /
  `create_mysql_engine`, `sender.Sender` (S3 client mocked), `spark_save_table`'s Delta
  write-chain (mock DataFrame), and `silver_data.read_sql_file` + the `.format()` brace-safety
  contract of every file in `src/queries/`.
- **`app/api/tests/`** — the FastAPI routes, model mocked via `main.model_find`.
- **`app/streamlit/tests/`** — the pandas-only helpers: `compute_*`, `format_color`, `_rank_by`,
  `_color_map`, `get_id_predictions`.

**Not** covered: the actual SQL transformations / Silver Spark execution (only the query files'
`.format()` safety is smoke-checked, not what they compute), the DAG, and the training script.
`app/api` and `app/streamlit` set `[tool.uv] package = false` (single-module services); their
`[dependency-groups].dev` adds `pytest` (and `httpx`, for the API's `TestClient`).

CI: `.github/workflows/tests.yml` runs on every `push` and `pull_request` — a `format` job
(`uvx ruff@0.16.2 format --check .`) plus a `pytest` matrix (one job per uv project,
`uv run --locked pytest -q`). `setup-uv` is pinned to `0.12.0` to match the root `uv_build`
constraint; ruff is pinned to `0.16.2` (matches the root dep). `ruff check` (lint) is **not** in
CI — the `# %%` script modules carry ~16 long-standing findings; only `ruff format` is kept
clean.

Each app has its own Dockerfile and is built independently by `docker-compose.yml`:
- `app/api` — FastAPI service, own `pyproject.toml`/`uv.lock`
- `app/streamlit` — Streamlit dashboard, own `pyproject.toml`/`uv.lock`

`.devcontainer/` (VS Code, "Python 3.13 and Java 17") is the one place the Spark/Delta stages
run without extra setup — it presets `JAVA_HOME` and `PYSPARK_SUBMIT_ARGS` and forwards Jupyter
(8888) and the Spark UI (4040). Outside it (or `docker compose`), the Bronze/Silver stages need
a local JVM + Java 17.

Root `main.py` and `src/lake_fastf1/__init__.py` (and the `lake-fastf1` console script) are
leftover `uv init` scaffolding — the real code is in `src/*.py`, `dags/`, and `app/`.

## Environment variables

Config is entirely via env vars (loaded from `.env`, which is gitignored — the tracked template is `.env.example`; copy it to `.env` and fill in the blanks). Every pipeline module (`src/*.py`) reads required paths via `os.environ[...]` and will raise `KeyError` if unset — always run through `docker compose` or with `.env` sourced.

Key vars: `PATH_RAW`, `PATH_BRONZE`, `PATH_SILVER`, `PATH_QUERIES` (data lake layer paths + SQL directory), `MLFLOW_URI`, `MLFLOW_MODEL_REGISTERED`, `MLFLOW_EXPERIMENT_NAME`, `API_PORT`, `MYSQL_HOST`/`MYSQL_PORT`/`MYSQL_USER`/`MYSQL_PASSWORD`/`MYSQL_ID_TABLE` (used by `src/sender_local.py` to mirror Bronze/Silver Delta tables into MySQL), `AWS_KEY`/`AWS_SECRET_KEY` (used by `src/sender.py` for S3 upload of raw Parquet files).

The Streamlit container additionally needs `TABLE_PATH_SILVER` and `TABLE_PATH_BRONZE` (paths as mounted read-only inside that container, not the same as `PATH_SILVER`/`PATH_BRONZE`) and reaches the API at `http://api-driver-champion:{API_PORT}` (Docker Compose service name), not localhost.

## Architecture

### Data pipeline layers

The DAG (`dags/data_pipeline.py`, `formula_one_data_pipeline`) wires these stages with Airflow `Asset`s as inlets/outlets, so lineage is explicit: `raw >> bronze >> silver >> sender_mysql`.

1. **Raw** (`src/extract_data.py`, `ExtractData`): pulls each year/round/session (identifiers `"R"` race, `"S"` sprint) from FastF1, flattens session results into a DataFrame, and writes one Parquet file per `{year}_{round:02}_{identifier}.parquet` under `PATH_RAW/results`. Skips files that already exist unless `reload_data=True` (which backfills from 1980 to present). Sleeps between requests to be polite to the FastF1/Ergast backend.

2. **Bronze** (`src/spark_session.py`, `consolidate_data`): reads all Raw Parquet files with Spark and writes a single coalesced Delta table at `PATH_BRONZE/results` (mode `overwrite`). `spark_session()` and `spark_save_table()` here are the shared Spark/Delta helpers used by the Silver layer too.

3. **Silver** (`src/silver_data.py`, `SilverData`): loads SQL from `PATH_QUERIES/*.sql` (via `src/queries/`) and runs it against Spark SQL temp views created from Bronze/Silver Delta tables. Produces:
   - `champions` (`champions.sql`) — one row per year: the points leader, `rank_driver`.
   - `driver_statistic_{5,10,20,40,50}` (`driver_statistic.sql`, parameterized by `{last_rounds}`) — rolling-window driver stats as of each `dt_ref` (one row per date a session happened), looking back N rounds, split into overall/Race/Sprint metrics. This is the most complex query in the repo — it builds a per-date "as of" rolling window from `results`, not simple groupby aggregates.
   - `driver_all_statistic` (`consolidate_drivers_statistic`) — inner-joins all five `driver_statistic_N` tables together on `(dt_ref, DriverId)`, suffixing every metric column with its window size (`_5`, `_10`, ...) via `add_suffix`.
   - `tb_abt` (`tb_abt.sql`) — the model's analytical base table: `driver_all_statistic` left-joined to `champions` to produce the `flChampion` label (was this driver champion that year, as of that `dt_ref`).

4. **MySQL mirror** (`src/sender_local.py`, `send_layer_to_mysql`): walks a layer's directory (Bronze or Silver) for `_delta_log` subfolders (i.e. every Delta table under that layer), reads each with Spark, converts to pandas, and replaces the corresponding MySQL table (named `{MYSQL_ID_TABLE}_{relative_path_with_underscores}`). This is how downstream/BI consumers outside the Spark/Delta world get the data.

5. **S3 upload** (`src/sender.py`, standalone CLI, not wired into the DAG): uploads all `*.parquet` files from a local folder to S3 and deletes them locally afterward — a separate archival path for Raw data, run manually via `python -m src.sender --bucket ...`.

All Silver SQL files are read as raw strings and `.format()`-ed (not parameterized queries) — `driver_statistic.sql` uses `{year_start}`, `{year_stop}`, `{last_rounds}` placeholders. If you edit these queries, curly braces anywhere in the SQL (e.g. in comments) will break `.format()`.

### Model training

`src/train_driver_champion.py` is a script-notebook (uses `# %%` cell markers, meant to be run interactively, e.g. in Jupyter/VS Code). It reads `tb_abt` from Silver, builds a time-based split (train < 2024, test == 2024, out-of-time == 2025), fits a `SimpleImputer` + `RandomForestClassifier` sklearn pipeline predicting `flChampion`, logs metrics/artifacts/model to MLflow, then retrains on the full dataset before registering. Feature columns are `df.iloc[:, 3:]` — i.e. everything after the first 3 columns (`dt_ref`, `DriverId`, label-adjacent columns) — so column order in `tb_abt.sql` matters.

### Serving layer

- **`app/api/main.py`** (FastAPI): loads the latest version of `MLFLOW_MODEL_REGISTERED` from the MLflow registry on every `/predict` call (no caching — `model_find` re-queries MLflow each request), predicts win probability, and returns a dict keyed by the caller-supplied `id`. Callers must include an `id` field per row in the request body; feature columns are selected via `model.feature_names_in_`, so the request payload must carry every feature the trained pipeline expects.

- **`app/streamlit/main.py`** (dashboard): reads Bronze (`get_bronze`) and Silver `tb_abt` (`get_predictions`) directly from the Delta tables via `deltalake.DeltaTable` (not Spark), calls the FastAPI `/predict` endpoint to get win probabilities for the Silver rows, and merges everything with driver metadata (team, color, headshot) pulled from Bronze. Both loaders are `st.cache_data(ttl="1d")`. Sections are: KPI cards, season snapshot, recent race cards, then tabs for win probability / points ranking / season progression / position heatmap / driver stats / constructors / raw data. See `README.md` for the full tab-by-tab layout description if extending the dashboard.

### Spark/Delta conventions

- `spark_save_table()` always does `.coalesce(1).write.format("delta").mode("overwrite")` — every Silver/Bronze table is a single-file full overwrite, not an incremental/append write. There is no partitioning or merge/upsert logic anywhere in the pipeline.
- Every `SilverData` operation opens its own `SparkSession` (`spark_session()` inside `__init__`); the DAG explicitly calls `.stop()` in a `finally` block after each task group's work, since Airflow tasks in this DAG run in-process (not via `SparkSubmitOperator`).
- Delta tables are read into Spark SQL via `createOrReplaceTempView`, so all Silver transformations are plain Spark SQL against `PATH_QUERIES/*.sql`, not DataFrame API chains.

### Airflow specifics

- The DAG uses the Airflow 3 SDK style (`from airflow.sdk import Asset, dag, task, task_group`), decorator-based tasks, and explicit `Asset` inlets/outlets for data lineage between Raw/Bronze/Silver/MySQL stages — not the classic `DAG()`/`PythonOperator` style.
- Scheduled weekly (`schedule="0 0 * * 1"`, i.e. Monday), `catchup=False`, `max_active_runs=1`.
- The Airflow image (root `Dockerfile`) installs `requirements.txt` via pip, not `uv sync` — the `uv`-based install is present but commented out. Keep `requirements.txt` and root `pyproject.toml` dependencies in sync when adding packages needed by DAG tasks.
