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


def build_caption(product: dict, include_brand: bool = True) -> str:
    """
    Constructs a natural language caption from cleaned product attributes.

    Captions are attribute-focused and query-styled to reduce the train/inference
    distribution gap. Raw titles are intentionally not used because they contain
    marketing noise, size codes, and language artifacts that conflict with the
    compositionality objective of the model.

    Args:
        product: A dict with cleaned product fields (as returned by Product.to_dict()).
        include_brand: When True, appends the brand as 'by <brand>'. Generating
                       captions with and without brand creates two distinct phrasings
                       per product, which prevents the model from overfitting to a
                       single rigid sentence structure.

    Returns:
        A short natural language string suitable for image-text contrastive training.

    Examples:
        build_caption({"color": "brown", "material": "suede", "product_type": "shoes",
                       "brand": "The Fix", "category": "Loafer"}, include_brand=True)
        -> "a brown suede shoes in the loafer category by The Fix"

        build_caption({"color": None, "material": None, "product_type": "grocery",
                       "brand": None, "category": None}, include_brand=False)
        -> "a grocery"
    """
    parts = []

    if product.get("color"):
        parts.append(product["color"].lower())

    if product.get("material"):
        parts.append(product["material"].lower())

    if product.get("product_type"):
        parts.append(product["product_type"].lower())

    category = product.get("category")
    product_type = product.get("product_type", "")
    if category and category.lower() != product_type.lower():
        parts.append(f"in the {category.lower()} category")

    body = " ".join(parts) if parts else "product"
    brand = product.get("brand")

    if include_brand and brand:
        return f"a {body} by {brand}"

    return f"a {body}"
