"""Structural tests for Delta writes without starting Spark."""

from unittest.mock import MagicMock

from src.spark_session import spark_save_table


def test_spark_save_table_does_not_force_a_single_partition():
    df = MagicMock(name="DataFrame")

    spark_save_table("/tmp/x/results", df)

    df.coalesce.assert_not_called()
    writer = df.write
    writer.format.assert_called_once_with("delta")
    writer.format.return_value.mode.assert_called_once_with("overwrite")
    writer.format.return_value.mode.return_value.option.assert_called_once_with(
        "overwriteSchema", "true"
    )
    writer.format.return_value.mode.return_value.option.return_value.save.assert_called_once_with(
        "/tmp/x/results"
    )


def test_spark_save_table_honors_an_explicit_partition_count():
    df = MagicMock(name="DataFrame")

    spark_save_table("/tmp/x/results", df, output_partitions=2)

    df.coalesce.assert_called_once_with(2)
    df.coalesce.return_value.write.format.assert_called_once_with("delta")
