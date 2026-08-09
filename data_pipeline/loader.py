"""
data_pipeline/loader.py

Provides class interfaces to load raw Amazon Berkeley Objects (ABO) files.
"""

import csv
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Generator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config

logger = logging.getLogger(__name__)


class ABOLoader:
    """
    Handles file-level loading of listing records and image mappings.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the loader with a Config instance.

        Args:
            config: The root project Configuration.
        """
        self._config = config

    def load_image_metadata(self) -> Dict[str, str]:
        """
        Loads the image CSV metadata mapping image ID to relative file path.

        Returns:
            A dictionary mapping image_id strings to relative image path strings.
        """
        logger.info("Loading image metadata from %s", self._config.images_meta_path)
        image_lookup: Dict[str, str] = {}

        with gzip.open(self._config.images_meta_path, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_lookup[row["image_id"]] = row["path"]

        logger.info("Successfully loaded %d image mappings", len(image_lookup))
        return image_lookup

    def iter_listings(self) -> Generator[dict, None, None]:
        """
        Iterates and streams raw JSON records from listings gzipped archives.

        Yields:
            A raw product dictionary from the listing files.
        """
        listing_files = sorted(self._config.listings_dir.glob("*.json.gz"))
        logger.info("Found %d listing archive files to process", len(listing_files))

        for file_path in listing_files:
            logger.info("Processing listing file: %s", file_path.name)
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line:
                        yield json.loads(stripped_line)
