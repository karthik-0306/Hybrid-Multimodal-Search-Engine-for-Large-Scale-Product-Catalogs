"""
scripts/fix_image_urls.py

One-off repair: rewrites the `image_url` payload on every point in the Qdrant
Cloud collection so the S3 region in the host is correct. Only touches points
whose URL is wrong — safe to re-run. Does NOT re-upload any vectors.

Usage:
    conda activate ai_env
    python scripts/fix_image_urls.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
BUCKET = os.getenv("AWS_BUCKET_NAME", "processed-abo-dataset")
# The bucket really lives in eu-north-1 (verified). Override via .env if needed.
REGION = os.getenv("AWS_REGION", "eu-north-1")

COLLECTION = "abo_products"
GOOD_HOST = f"{BUCKET}.s3.{REGION}.amazonaws.com"


def corrected(url: str):
    """Return a fixed URL, or None if it is already correct / not fixable."""
    if not url or GOOD_HOST in url:
        return None
    marker = ".amazonaws.com/"
    if marker not in url:
        return None
    tail = url.split(marker, 1)[1]          # e.g. "images/78449e17.jpg"
    return f"https://{GOOD_HOST}/{tail}"


def main() -> None:
    print(f"Target host: https://{GOOD_HOST}/...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    offset = None
    scanned = fixed = 0
    while True:
        points, offset = client.scroll(
            COLLECTION,
            limit=2000,
            offset=offset,
            with_payload=["image_url"],
            with_vectors=False,
        )
        for p in points:
            scanned += 1
            new_url = corrected((p.payload or {}).get("image_url", ""))
            if not new_url:
                continue
            for attempt in range(5):
                try:
                    client.set_payload(
                        collection_name=COLLECTION,
                        payload={"image_url": new_url},
                        points=[p.id],
                        wait=False,
                    )
                    fixed += 1
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
        print(f"  scanned {scanned}, fixed {fixed}", flush=True)
        if offset is None:
            break

    print(f"Done. Scanned {scanned} points, fixed {fixed}.")


if __name__ == "__main__":
    main()
