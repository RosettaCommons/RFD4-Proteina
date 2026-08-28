#!/usr/bin/env python3
"""Fetch one object from an S3-compatible bucket via boto3's multipart-parallel downloader.

Used by download_example_data.sh in place of `aws s3 cp` -- same S3 multipart-GET speed, no
aws-cli install (boto3 is a normal Python dependency, already in the venv via `.[dev]`).

Usage::

    python scripts/s3_fetch.py <bucket> <key> <dest> <endpoint_url>
"""

import boto3
import fire
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config


def main(bucket: str, key: str, dest: str, endpoint_url: str) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        config=Config(signature_version=UNSIGNED, s3={"addressing_style": "virtual"}),
    )
    transfer_config = TransferConfig(max_concurrency=20, multipart_chunksize=64 * 1024 * 1024)
    client.download_file(bucket, key, dest, Config=transfer_config)


if __name__ == "__main__":
    fire.Fire(main)
