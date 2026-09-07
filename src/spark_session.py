# %%
import os

from delta import configure_spark_with_delta_pip
from pyspark.sql import DataFrame, SparkSession

PATH_RAW = os.environ["PATH_RAW"]
PATH_BRONZE = os.environ["PATH_BRONZE"]


def spark_session():
    builder = (
        SparkSession.builder.appName("PySpark")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def spark_save_table(
    path: str, df: DataFrame, output_partitions: int | None = None
) -> None:
    """Sobrescreve uma tabela Delta sem forçar um gargalo de partição única."""
    output = df.coalesce(output_partitions) if output_partitions is not None else df
    (
        output.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(path)
    )


def consolidate_data(data_set: str = "results"):
    spark = spark_session()
    try:
        df = spark.read.format("parquet").load(f"{PATH_RAW}/{data_set}/*.parquet")
        spark_save_table(f"{PATH_BRONZE}/{data_set}", df)
    finally:
        spark.stop()


def main():
    consolidate_data("results")


# %%
if __name__ == "__main__":
    main()

# %%
