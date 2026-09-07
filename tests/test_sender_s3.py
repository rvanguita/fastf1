"""Tests for ``src.sender.Sender`` — the S3 client is always a mock."""

from unittest.mock import Mock

import pytest
from boto3.exceptions import S3UploadFailedError

import src.sender as sender_mod
from src.sender import Sender


@pytest.fixture
def sender(monkeypatch):
    fake_client = Mock()
    monkeypatch.setattr(sender_mod.boto3, "client", lambda *a, **k: fake_client)
    s = Sender("my-bucket", "results")
    s.s3 = fake_client
    return s


class TestProcessFile:
    def test_uploads_then_deletes_local_file(self, sender, tmp_path):
        f = tmp_path / "2024_01_R.parquet"
        f.write_bytes(b"data")

        assert sender.process_file(str(f)) is True
        sender.s3.upload_file.assert_called_once_with(
            str(f), "my-bucket", "results/2024_01_R.parquet"
        )
        assert not f.exists()

    def test_returns_false_and_keeps_file_on_error(self, sender, tmp_path):
        f = tmp_path / "2024_02_R.parquet"
        f.write_bytes(b"data")
        sender.s3.upload_file.side_effect = S3UploadFailedError("network down")

        assert sender.process_file(str(f)) is False
        assert f.exists()


class TestProcessFolder:
    def test_only_parquet_files_are_processed(self, sender, tmp_path):
        for name in ("a.parquet", "b.parquet", "c.txt", "d.csv"):
            (tmp_path / name).write_bytes(b"x")

        seen = []
        sender.process_file = lambda path: seen.append(path.split("/")[-1])
        sender.process_folder(str(tmp_path))

        assert sorted(seen) == ["a.parquet", "b.parquet"]
