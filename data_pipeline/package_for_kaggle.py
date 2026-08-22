"""
data_pipeline/package_for_kaggle.py

Creates a self-contained zip archive containing:
  - products.parquet (the cleaned product catalogue)
  - All 56,427 product images referenced in products.parquet

This archive is uploaded to Kaggle as a Dataset so the fine-tuning notebook
can access both the catalogue and the images without additional setup.

Usage:
    python data_pipeline/package_for_kaggle.py

Output:
    kaggle_dataset.zip in the project root directory.
"""

import logging
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_ZIP_NAME = "kaggle_dataset.zip"


def package_dataset(config: Config) -> None:
    """
    Assembles the Kaggle upload archive.

    Only images referenced in products.parquet are included. The full image
    catalogue (398,212 files) is much larger than needed — including only the
    56,427 used images keeps the upload manageable.

    The archive layout inside the zip:
        products.parquet
        images/<original relative path>   (e.g. images/3a/3a4e88ef.jpg)

    Args:
        config: Root config definitions containing processed and raw data paths.
    """
    output_path = config.project_root / OUTPUT_ZIP_NAME

    logger.info("Loading cleaned product catalogue from %s", config.products_parquet)
    df = pd.read_parquet(config.products_parquet)
    logger.info("Catalogue contains %d products", len(df))

    # Deduplicate image paths to prevent warnings and redundant zip entries
    unique_image_paths = sorted(list(set(df["main_image_path"].dropna().tolist())))
    logger.info("Identified %d unique active images to package", len(unique_image_paths))

    logger.info("Creating archive: %s", output_path)
    missing_count = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add the products parquet first
        zf.write(config.products_parquet, arcname="products.parquet")
        logger.info("Added products.parquet to archive")

        # Add each referenced image
        for rel_path in unique_image_paths:
            abs_path = config.processed_images_dir / rel_path
            if not abs_path.exists():
                logger.warning("Image not found, skipping: %s", abs_path)
                missing_count += 1
                continue
            # Store under images/<rel_path> to keep a flat, predictable layout
            zf.write(abs_path, arcname=f"images/{rel_path}")

    added_images = len(unique_image_paths) - missing_count
    size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info("Archive complete: %s", output_path)
    logger.info("Unique images added: %d | Images missing: %d", added_images, missing_count)
    logger.info("Archive size: %.1f MB", size_mb)


def main() -> None:
    """
    Entry point for the Kaggle dataset packaging script.
    """
    project_root = Path(__file__).resolve().parent.parent
    config = Config(project_root)

    if not config.products_parquet.exists():
        raise FileNotFoundError(
            f"products.parquet not found at {config.products_parquet}. "
            "Run data_pipeline/clean.py first."
        )

    package_dataset(config)


if __name__ == "__main__":
    main()
