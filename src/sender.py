# %%

import argparse
import logging
import os
import posixpath
from pathlib import Path

import boto3
import dotenv
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import BotoCoreError, ClientError
from rich.progress import track

dotenv.load_dotenv()

AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
PATH_RAW = os.getenv("PATH_RAW")
REGION_NAME = os.getenv("REGION_NAME", "us-east-1")
LOGGER = logging.getLogger(__name__)


class Sender:
    def __init__(self, bucket_name: str, bucket_folder: str) -> None:
        self.bucket_name = bucket_name
        self.bucket_folder = bucket_folder

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=REGION_NAME,
        )

    def process_file(self, filename: str) -> bool:
        file = Path(filename).name
        bucket_path = posixpath.join(self.bucket_folder, file)

        try:
            self.s3.upload_file(filename, self.bucket_name, bucket_path)
        except (BotoCoreError, ClientError, S3UploadFailedError, OSError) as err:
            LOGGER.error(
                "Falha ao enviar %s para s3://%s/%s: %s",
                filename,
                self.bucket_name,
                bucket_path,
                err,
            )
            return False

        os.remove(filename)
        return True

    def process_folder(self, folder: str) -> None:
        files = [i for i in os.listdir(folder) if i.endswith(".parquet")]
        for f in track(files):
            self.process_file(os.path.join(folder, f))


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--bucket_path", default="results", type=str)
    parser.add_argument("--folder", default=PATH_RAW, type=str)
    args = parser.parse_args()

    if args.bucket:
        send = Sender(args.bucket, args.bucket_path)
        send.process_folder(args.folder)
