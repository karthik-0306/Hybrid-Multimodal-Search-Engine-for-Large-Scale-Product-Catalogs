"""
data_pipeline/download.py

Verifies the presence and integrity of downloaded raw ABO files.
Since the data is already stored locally, this module serves as a checks-and-balances
validation script to confirm raw directories are correctly populated.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def verify_raw_data(config: Config) -> bool:
    """
    Verifies that all required raw data assets exist on the local file system.

    Args:
        config: Root config definitions containing raw paths.

    Returns:
        True if all files exist, False otherwise.
    """
    success = True

    # 1. Verify images csv metadata
    if not config.images_meta_path.exists():
        logger.error("Missing images metadata file: %s", config.images_meta_path)
        success = False
    else:
        logger.info("Found images metadata: %s", config.images_meta_path.name)

    # 2. Verify listings metadata directory and files
    if not config.listings_dir.exists():
        logger.error("Missing listings metadata directory: %s", config.listings_dir)
        success = False
    else:
        listing_files = list(config.listings_dir.glob("*.json.gz"))
        if not listing_files:
            logger.error("No .json.gz listing archives found in %s", config.listings_dir)
            success = False
        else:
            logger.info("Found %d raw listing archive files", len(listing_files))

    # 3. Verify images folder structures
    if not config.images_base_dir.exists():
        logger.error("Missing images directory: %s", config.images_base_dir)
        success = False
    else:
        logger.info("Found images small directory structure")

    return success


def main() -> None:
    """
    Main entry point for raw dataset verification.
    """
    project_root = Path(__file__).resolve().parent.parent
    config = Config(project_root)

    logger.info("Starting verification of raw ABO dataset assets")
    is_valid = verify_raw_data(config)

    if is_valid:
        logger.info("Verification complete: Raw data exists and is mapped correctly")
    else:
        logger.error("Verification failed: Raw data package is incomplete or misplaced")
        raise FileNotFoundError("Raw data assets missing from project paths")


if __name__ == "__main__":
    main()
