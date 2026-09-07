"""Airflow DAG for the Formula 1 data pipeline.

This workflow orchestrates the full ETL journey from raw FastF1 data
extraction to Bronze and Silver transformations. It is designed as a
long-term pipeline that will continue to evolve as data quality,
performance, and analytical requirements improve.
"""

import os
from datetime import datetime

from airflow.sdk import (
    Asset,
    dag,
    task,
    task_group,
)

from src.extract_data import ExtractData
from src.sender_local import send_layer_to_mysql
from src.silver_data import SilverData
from src.spark_session import consolidate_data

PATH_RAW = os.environ["PATH_RAW"]
PATH_BRONZE = os.environ["PATH_BRONZE"]
PATH_SILVER = os.environ["PATH_SILVER"]


# ---------------------------------------------------------------------------
# Raw assets
# ---------------------------------------------------------------------------

RAW_RESULTS = Asset(
    name="raw_results",
    uri=f"file://{PATH_RAW}/results",
)


# ---------------------------------------------------------------------------
# Bronze assets
# ---------------------------------------------------------------------------

BRONZE_RESULTS = Asset(
    name="bronze_results",
    uri=f"file://{PATH_BRONZE}/results",
)


# ---------------------------------------------------------------------------
# Silver assets
# ---------------------------------------------------------------------------

SILVER_CHAMPIONS = Asset(
    name="silver_champions",
    uri=f"file://{PATH_SILVER}/champions",
)

SILVER_DRIVER_STATISTIC_5 = Asset(
    name="silver_driver_statistic_5",
    uri=f"file://{PATH_SILVER}/driver_statistic_5",
)

SILVER_DRIVER_STATISTIC_10 = Asset(
    name="silver_driver_statistic_10",
    uri=f"file://{PATH_SILVER}/driver_statistic_10",
)

SILVER_DRIVER_STATISTIC_20 = Asset(
    name="silver_driver_statistic_20",
    uri=f"file://{PATH_SILVER}/driver_statistic_20",
)

SILVER_DRIVER_STATISTIC_40 = Asset(
    name="silver_driver_statistic_40",
    uri=f"file://{PATH_SILVER}/driver_statistic_40",
)

SILVER_DRIVER_STATISTIC_50 = Asset(
    name="silver_driver_statistic_50",
    uri=f"file://{PATH_SILVER}/driver_statistic_50",
)

SILVER_DRIVER_ALL_STATISTIC = Asset(
    name="silver_driver_all_statistic",
    uri=f"file://{PATH_SILVER}/driver_all_statistic",
)

SILVER_TB_ABT = Asset(
    name="silver_tb_abt",
    uri=f"file://{PATH_SILVER}/tb_abt",
)

SILVER_MART_DRIVER_ROUND = Asset(
    name="silver_mart_driver_round",
    uri=f"file://{PATH_SILVER}/mart_driver_round",
)

SILVER_MART_STANDINGS = Asset(
    name="silver_mart_standings",
    uri=f"file://{PATH_SILVER}/mart_standings",
)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


@dag(
    dag_id="data-pipeline",
    description="Raw, Bronze, and Silver data pipeline",
    schedule="0 0 * * 1",
    start_date=datetime(2026, 7, 28),
    catchup=False,
    max_active_runs=1,
    tags=["f1", "etl"],
)
def formula_one_data_pipeline() -> None:

    # -----------------------------------------------------------------------
    # Raw
    # -----------------------------------------------------------------------

    @task_group(group_id="raw")
    def raw_layer() -> None:

        @task(
            task_id="extract_fastf1",
            outlets=[RAW_RESULTS],
        )
        def extract_fastf1() -> bool:
            extractor = ExtractData()
            return extractor.process_years()

        # @task.short_circuit(task_id="has_new_data")
        # def has_new_data(updated: bool) -> bool:
        #     return updated

        extracted = extract_fastf1()
        # has_new_data(extracted)

    # -----------------------------------------------------------------------
    # Bronze
    # -----------------------------------------------------------------------

    @task_group(group_id="bronze")
    def bronze_layer() -> None:

        @task(
            task_id="consolidated_results",
            inlets=[RAW_RESULTS],
            outlets=[BRONZE_RESULTS],
        )
        def create_results() -> None:
            consolidate_data()

        create_results()

    # -----------------------------------------------------------------------
    # Silver
    # -----------------------------------------------------------------------

    @task_group(group_id="silver")
    def silver_layer() -> None:

        # -------------------------------------------------------------------
        # Silver: Champions
        # -------------------------------------------------------------------

        @task_group(group_id="champions")
        def champions_group() -> None:

            @task(
                task_id="create",
                inlets=[BRONZE_RESULTS],
                outlets=[SILVER_CHAMPIONS],
            )
            def create_champions() -> None:
                silver_data = SilverData()
                try:
                    (silver_data.read_save_query("champions"))
                finally:
                    silver_data.stop()

            create_champions()

        # -------------------------------------------------------------------
        # Silver: Driver statistics
        # -------------------------------------------------------------------

        @task_group(group_id="driver_statistics")
        def driver_statistics_group() -> None:

            @task(
                task_id="rounds_driver_statistic",
                inlets=[BRONZE_RESULTS],
                outlets=[
                    SILVER_DRIVER_STATISTIC_5,
                    SILVER_DRIVER_STATISTIC_10,
                    SILVER_DRIVER_STATISTIC_20,
                    SILVER_DRIVER_STATISTIC_40,
                    SILVER_DRIVER_STATISTIC_50,
                ],
            )
            def create_driver_statistic() -> None:
                rounds = [5, 10, 20, 40, 50]
                for round in rounds:
                    silver_data = SilverData()
                    try:
                        query_name = "driver_statistic"

                        silver_data.driver_n_race(query_name=query_name, round=round)
                    finally:
                        silver_data.stop()

            create_driver_statistic()

        # -------------------------------------------------------------------
        # Silver: Consolidated driver statistics
        # -------------------------------------------------------------------

        @task_group(group_id="driver_all_statistic")
        def driver_consolidate_statistic() -> None:

            @task(
                task_id="consolidate",
                inlets=[
                    SILVER_DRIVER_STATISTIC_5,
                    SILVER_DRIVER_STATISTIC_10,
                    SILVER_DRIVER_STATISTIC_20,
                    SILVER_DRIVER_STATISTIC_40,
                    SILVER_DRIVER_STATISTIC_50,
                ],
                outlets=[SILVER_DRIVER_ALL_STATISTIC],
            )
            def consolidate_driver_statistics() -> None:
                silver_data = SilverData()
                try:
                    (silver_data.consolidate_drivers_statistic())
                finally:
                    silver_data.stop()

            consolidate_driver_statistics()

        # -------------------------------------------------------------------
        # Silver: ABT
        # -------------------------------------------------------------------

        @task_group(group_id="abt")
        def abt_group() -> None:

            @task(
                task_id="create",
                inlets=[
                    SILVER_CHAMPIONS,
                    SILVER_DRIVER_ALL_STATISTIC,
                ],
                outlets=[SILVER_TB_ABT],
            )
            def create_abt() -> None:
                silver_data = SilverData()
                try:
                    (silver_data.tb_abt())
                finally:
                    silver_data.stop()

            create_abt()

        @task_group(group_id="analytical_marts")
        def analytical_marts_group() -> None:

            @task(
                task_id="create",
                inlets=[BRONZE_RESULTS],
                outlets=[SILVER_MART_DRIVER_ROUND, SILVER_MART_STANDINGS],
            )
            def create_analytical_marts() -> None:
                silver_data = SilverData()
                try:
                    silver_data.analytical_marts()
                finally:
                    silver_data.stop()

            create_analytical_marts()

        champions = champions_group()
        drivers_statistics_group = driver_statistics_group()
        driver_all_statistic = driver_consolidate_statistic()
        abt = abt_group()
        analytical_marts = analytical_marts_group()

        champions
        drivers_statistics_group >> driver_all_statistic
        [champions, driver_all_statistic] >> abt
        analytical_marts

    @task(task_id="sender_mysql")
    def sender_mysql() -> None:
        layers = {
            "bronze": PATH_BRONZE,
            "silver": PATH_SILVER,
        }

        for layer, path in layers.items():
            send_layer_to_mysql(
                layer=layer,
                root_path=path,
            )

    # -----------------------------------------------------------------------
    # Main flow
    # -----------------------------------------------------------------------

    raw = raw_layer()
    bronze = bronze_layer()
    silver = silver_layer()
    sender_data = sender_mysql()

    raw >> bronze >> silver >> sender_data


f1_data_pipeline = formula_one_data_pipeline()
