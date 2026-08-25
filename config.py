"""
config.py

This module serves as the single source of truth for all paths, constants,
and configuration parameters across the project.
"""

from pathlib import Path
from typing import Dict, Tuple


class Config:
    """
    Configuration manager for the search engine project.

    All file paths are resolved relative to a given project root directory to
    ensure compatibility across different development environments.
    """

    # Model Configuration
    BASE_MODEL_NAME: str = "google/siglip2-base-patch16-224"

    # Data Processing Constraints
    DROPPED_PRODUCT_TYPES: Tuple[str, ...] = ("cellular_phone_case",)
    LANGUAGE_PRIORITY: Tuple[str, ...] = ("en_US", "en_CA", "en_GB", "en_AU", "en_IN")
    COUNTRY_PRIORITY: Dict[str, int] = {
        "US": 1,
        "CA": 2,
        "GB": 3,
        "AU": 4,
        "IN": 5,
    }

    def __init__(self, project_root: Path) -> None:
        """
        Initializes path configurations relative to the project root.

        Args:
            project_root: The root directory of the repository.
        """
        self.project_root = project_root

        # Raw Data Paths
        self.raw_data_dir: Path = project_root / "Raw_Data"
        self.listings_dir: Path = self.raw_data_dir / "abo-listings" / "listings" / "metadata"
        self.images_meta_path: Path = (
            self.raw_data_dir
            / "abo-images-small"
            / "images"
            / "metadata"
            / "images.csv.gz"
        )
        self.images_base_dir: Path = (
            self.raw_data_dir / "abo-images-small" / "images" / "small"
        )

        # Processed Data Paths
        self.processed_dir: Path = project_root / "Processed_Data"
        self.products_parquet: Path = self.processed_dir / "products.parquet"
        self.images_meta_parquet: Path = self.processed_dir / "images_meta.parquet"
        self.processed_images_dir: Path = self.processed_dir / "images"
        self.lora_adapter_dir: Path = self.processed_dir / "models" / "lora_adapter"
        self.image_embeddings_npy: Path = self.processed_dir / "image_embeddings.npy"
        self.qdrant_db_dir: Path = self.processed_dir / "qdrant_db"
        self.stats_json: Path = self.processed_dir / "stats" / "dataset_stats.json"

    def __repr__(self) -> str:
        """
        Returns a string representation of the Config instance.

        Returns:
            A string containing the project root path.
        """
        return f"Config(project_root={self.project_root})"
