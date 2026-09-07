# %%
import os
from datetime import UTC, datetime
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.spark_session import spark_save_table, spark_session

CURRENT_YEAR = datetime.now(UTC).year

JOIN_COLUMNS = ["dt_ref", "DriverId"]
PATH_QUERIES = os.environ["PATH_QUERIES"]
PATH_SILVER = os.environ["PATH_SILVER"]
PATH_BRONZE = os.environ["PATH_BRONZE"]


class SilverData:
    def __init__(self) -> None:
        self.spark = spark_session()
        self.spark_view_table(f"{PATH_BRONZE}/results", "results")

    def stop(self):
        self.spark.stop()

    def spark_view_table(self, path, table_name):
        table_view = self.spark.read.format("delta").load(path)
        table_view.createOrReplaceTempView(table_name)

    def read_save_query(self, query_name: str) -> None:
        query = self.read_sql_file(query_name)
        df = self.spark.sql(query)

        spark_save_table(f"{PATH_SILVER}/{query_name}", df)

    def driver_n_race(self, query_name: str, round: int) -> None:
        query = self.read_sql_file(query_name)
        df = self.spark.sql(
            query.format(year_start=1980, year_stop=CURRENT_YEAR, last_rounds=round)
        )

        spark_save_table(f"{PATH_SILVER}/{query_name}_{round}", df)

    def consolidate_drivers_statistic(
        self,
        rounds: tuple[int, ...] = (5, 10, 20, 40, 50),
        table_name: str = "driver_all_statistic",
    ) -> None:
        df_all = []

        for round in rounds:
            self.spark_view_table(
                f"{PATH_SILVER}/driver_statistic_{round}", f"driver_statistic_{round}"
            )

            df = self.spark.table(f"driver_statistic_{round}")
            df_all.append(
                self.add_suffix(
                    dataframe=df,
                    suffix=round,
                    join_columns=JOIN_COLUMNS,
                )
            )

        driver_features = reduce(
            lambda left, right: left.join(
                right,
                on=JOIN_COLUMNS,
                how="inner",
            ),
            df_all,
        )
        spark_save_table(f"{PATH_SILVER}/{table_name}", driver_features)

    def add_suffix(
        self,
        dataframe: DataFrame,
        suffix: str,
        join_columns: list[str],
    ) -> DataFrame:

        metric_columns = [
            F.col(column).alias(f"{column}_{suffix}")
            for column in dataframe.columns
            if column not in join_columns
        ]

        return dataframe.select(
            *[F.col(column) for column in join_columns],
            *metric_columns,
        )

    def tb_abt(self):
        champions = "champions"
        drivers = "driver_all_statistic"
        self.spark_view_table(f"{PATH_SILVER}/{champions}", f"{champions}")
        self.spark_view_table(f"{PATH_SILVER}/{drivers}", f"{drivers}")
        self.read_save_query("tb_abt")

    def analytical_marts(self) -> None:
        """Materializa os contratos analíticos consumidos por produtos de BI."""
        self.read_save_query("mart_driver_round")
        self.spark_view_table(f"{PATH_SILVER}/mart_driver_round", "mart_driver_round")
        self.read_save_query("mart_standings")
        self.validate_analytical_marts()

    def validate_analytical_marts(self) -> None:
        """Falha o pipeline quando chaves ou domínios dos marts são inválidos."""
        contracts = {
            "mart_driver_round": (["season", "round_number", "driver_id"], "points"),
            "mart_standings": (
                ["season", "round_number", "driver_id"],
                "cumulative_points",
            ),
        }
        for table_name, (keys, non_negative) in contracts.items():
            frame = self.spark.read.format("delta").load(f"{PATH_SILVER}/{table_name}")
            null_condition = reduce(
                lambda left, right: left | right,
                [F.col(column).isNull() for column in keys],
            )
            has_null_key = frame.filter(null_condition).limit(1).count() > 0
            has_duplicate_key = (
                frame.groupBy(*keys).count().filter(F.col("count") > 1).limit(1).count()
                > 0
            )
            has_negative_metric = (
                frame.filter(F.col(non_negative) < 0).limit(1).count() > 0
            )
            if has_null_key or has_duplicate_key or has_negative_metric:
                raise ValueError(
                    f"Data contract failed for {table_name}: "
                    f"null_key={has_null_key}, duplicate_key={has_duplicate_key}, "
                    f"negative_{non_negative}={has_negative_metric}"
                )

    def read_sql_file(self, query_name):
        with open(f"{PATH_QUERIES}/{query_name}.sql", "r") as file:
            query = file.read()
        return query


def main():
    silver_data = SilverData()

    silver_data.read_save_query("champions")

    # silver_data = SilverData()
    query_name = "driver_statistic"
    rounds = [5, 10, 20, 40, 50]

    # silver_data.sessions_last_n_race(query_name, rounds)
    # silver_data = SilverData()
    silver_data.driver_n_race(query_name, rounds[0])
    silver_data.driver_n_race(query_name, rounds[1])
    silver_data.driver_n_race(query_name, rounds[2])
    silver_data.driver_n_race(query_name, rounds[3])
    silver_data.driver_n_race(query_name, rounds[4])

    # silver_data = SilverData()
    (silver_data.consolidate_drivers_statistic())

    # silver_data = SilverData()
    silver_data.tb_abt()
    silver_data.analytical_marts()


# %%
if __name__ == "__main__":
    main()
# %%
