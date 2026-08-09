"""
data_pipeline/clean.py

Cleans, filters, and deduplicates the raw ABO listings and joins them with the
image metadata. Produces final parquet outputs.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from data_pipeline.loader import ABOLoader
from data_pipeline.schema import Product

# Configure logging format according to guidelines
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ABOProcessor:
    """
    Orchestrates the cleaning and validation pipeline for raw product catalog data.
    """

    def __init__(self, config: Config, loader: ABOLoader) -> None:
        """
        Initializes the processor with configuration and loader instances.

        Args:
            config: Root config definitions.
            loader: Dedicated loader for reading files.
        """
        self._config = config
        self._loader = loader

    def _get_english_value(
        self, field_list: list, use_standardized: bool = False
    ) -> Optional[str]:
        """
        Helper method to retrieve the best English value from a multilingual raw list.

        Args:
            field_list: List of language dictionary records.
            use_standardized: Whether to prefer standardized values.

        Returns:
            The chosen English string, or None if no English value is found.
        """
        if not isinstance(field_list, list) or not field_list:
            return None

        val_map: Dict[str, str] = {}
        for item in field_list:
            lang = item.get("language_tag", "")
            raw_val = item.get("value")

            if use_standardized:
                std_vals = item.get("standardized_values")
                if std_vals and isinstance(std_vals, list) and std_vals[0]:
                    val_map[lang] = str(std_vals[0])
                    continue

            if raw_val is not None:
                val_map[lang] = str(raw_val)

        for lang in self._config.LANGUAGE_PRIORITY:
            if lang in val_map:
                return val_map[lang]

        for lang, val in val_map.items():
            if lang.startswith("en_"):
                return val

        return None

    def _extract_leaf_category(self, nodes: list) -> Optional[str]:
        """
        Extracts the leaf category segment from the nodes metadata.

        Args:
            nodes: List of category node records.

        Returns:
            A string representing the leaf category, or None.
        """
        if not nodes:
            return None
        first = nodes[0]
        node_name = first.get("node_name", "") or first.get("path", "")
        if not node_name:
            return None
        segments = node_name.strip("/").split("/")
        return segments[-1] if segments else None

    def _clean_product_type(self, record: dict) -> Optional[str]:
        """
        Safely extracts and lowercases the product type.

        Args:
            record: Raw product record.

        Returns:
            Lowercased product type string, or None.
        """
        pt_data = record.get("product_type")
        if not pt_data:
            return None

        if isinstance(pt_data, list):
            if pt_data and isinstance(pt_data[0], dict):
                val = pt_data[0].get("value")
            elif pt_data:
                val = pt_data[0]
            else:
                val = None
        else:
            val = pt_data

        return str(val).lower() if val else None

    def _has_swatch(self, record: dict) -> bool:
        """
        Checks if the item name/title contains the word 'swatch'.

        Args:
            record: Raw product record.

        Returns:
            True if any item name contains 'swatch' case-insensitively, else False.
        """
        names = record.get("item_name", [])
        if not isinstance(names, list):
            return False

        for n in names:
            val = n.get("value", "")
            if val and "swatch" in str(val).lower():
                return True
        return False

    def _process_single_record(
        self, record: dict, image_lookup: Dict[str, str]
    ) -> Optional[Product]:
        """
        Applies cleaning, extracting, and validation filters on a raw record.

        Args:
            record: A single raw product record.
            image_lookup: Mappings of image ID to paths.

        Returns:
            A validated Product instance, or None if the record is dropped.
        """
        # Drop if it is a swatch product
        if self._has_swatch(record):
            return None

        # Clean and filter product type
        pt_cleaned = self._clean_product_type(record)
        if not pt_cleaned or pt_cleaned in self._config.DROPPED_PRODUCT_TYPES:
            return None

        # Clean and filter English title
        title = self._get_english_value(record.get("item_name", []))
        if not title:
            return None

        # Join image ID to relative path lookup
        main_image_id = record.get("main_image_id")
        if not main_image_id or main_image_id not in image_lookup:
            return None

        # Extract material with fabric_type fallback
        material = self._get_english_value(record.get("material", []))
        if not material:
            material = self._get_english_value(record.get("fabric_type", []))

        # Build clean Product dataclass matching target schema
        return Product(
            item_id=record["item_id"],
            product_type=pt_cleaned,
            title=title,
            color=self._get_english_value(
                record.get("color", []), use_standardized=True
            ),
            material=material,
            brand=self._get_english_value(record.get("brand", [])),
            category=self._extract_leaf_category(record.get("node", [])),
            main_image_id=main_image_id,
            main_image_path=image_lookup[main_image_id],
        )

    def clean_dataset(self) -> Tuple[List[Product], dict]:
        """
        Loads, cleans, deduplicates, and validates listings and image metadata.

        Returns:
            A tuple of cleaned Product records and execution statistics.
        """
        image_lookup = self._loader.load_image_metadata()

        # Deduplication lookup mapping: item_id -> (Product, country_priority_score)
        unique_products: Dict[str, Tuple[Product, int]] = {}

        total_raw = 0
        dropped_swatch = 0
        dropped_type = 0
        dropped_no_english = 0
        dropped_no_image = 0
        dropped_duplicate = 0

        for record in self._loader.iter_listings():
            total_raw += 1

            if self._has_swatch(record):
                dropped_swatch += 1
                continue

            pt_cleaned = self._clean_product_type(record)
            if not pt_cleaned or pt_cleaned in self._config.DROPPED_PRODUCT_TYPES:
                dropped_type += 1
                continue

            title = self._get_english_value(record.get("item_name", []))
            if not title:
                dropped_no_english += 1
                continue

            main_image_id = record.get("main_image_id")
            if not main_image_id or main_image_id not in image_lookup:
                dropped_no_image += 1
                continue

            # Process record to build Product
            product = self._process_single_record(record, image_lookup)
            if not product:
                continue

            country = record.get("country", "XX")
            priority = self._config.COUNTRY_PRIORITY.get(country, 99)

            if product.item_id in unique_products:
                _, existing_priority = unique_products[product.item_id]
                if priority >= existing_priority:
                    dropped_duplicate += 1
                    continue
                dropped_duplicate += 1

            unique_products[product.item_id] = (product, priority)

        final_products = [prod for prod, _ in unique_products.values()]

        stats = {
            "total_raw": total_raw,
            "dropped_swatch": dropped_swatch,
            "dropped_product_type": dropped_type,
            "dropped_no_english": dropped_no_english,
            "dropped_no_image": dropped_no_image,
            "dropped_duplicate": dropped_duplicate,
            "kept": len(final_products),
        }

        return final_products, stats


def main() -> None:
    """
    Main orchestrator function for the clean script.
    """
    project_root = Path(__file__).resolve().parent.parent
    config = Config(project_root)

    # Ensure output dirs exist
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.stats_json.parent.mkdir(parents=True, exist_ok=True)

    loader = ABOLoader(config)
    processor = ABOProcessor(config, loader)

    logger.info("Starting cleaning and filtering ETL run")
    products, stats = processor.clean_dataset()

    logger.info("Cleaning pipeline finished. Reporting stats:")
    logger.info("Total listings parsed: %d", stats["total_raw"])
    logger.info("Dropped (swatch titles): %d", stats["dropped_swatch"])
    logger.info("Dropped (unsupported type): %d", stats["dropped_product_type"])
    logger.info("Dropped (no English translation): %d", stats["dropped_no_english"])
    logger.info("Dropped (no valid image ID): %d", stats["dropped_no_image"])
    logger.info("Dropped (cross-marketplace duplicate): %d", stats["dropped_duplicate"])
    logger.info("Unique final records kept: %d", stats["kept"])

    logger.info("Saving products database to parquet: %s", config.products_parquet)
    df_products = pd.DataFrame([prod.to_dict() for prod in products])
    df_products.to_parquet(config.products_parquet, index=False)

    logger.info("Converting full images metadata to parquet")
    df_images = pd.read_csv(config.images_meta_path)
    df_images.to_parquet(config.images_meta_parquet, index=False)

    logger.info("Saving processing statistics report")
    stats["total_images"] = len(df_images)
    with open(config.stats_json, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("ETL run finished successfully")


if __name__ == "__main__":
    main()
