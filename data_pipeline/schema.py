"""
data_pipeline/schema.py

Defines the data contract for cleaned product listings.
"""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class Product:
    """
    Represents a single cleaned and normalized product listing.

    Attributes:
        item_id: Unique ASIN identifier for the product.
        product_type: Lowercased compositional attribute.
        title: First available English name variant.
        color: Standardized color value, falling back to raw color name.
        material: Primary material name, falling back to fabric type.
        brand: Brand name in English.
        category: Leaf segment name of the category node.
        main_image_id: Unique identifier for the primary product image.
        main_image_path: Relative path to the image JPEG file.
    """

    item_id: str
    product_type: str
    title: str
    color: Optional[str]
    material: Optional[str]
    brand: Optional[str]
    category: Optional[str]
    main_image_id: str
    main_image_path: str

    def to_dict(self) -> dict:
        """
        Converts the Product instance into a dictionary.

        Returns:
            A dictionary containing product attributes.
        """
        return asdict(self)
