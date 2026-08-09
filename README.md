# Hybrid Multimodal Search Engine for Large Scale Product Catalogs

This repository contains a production-ready hybrid search engine that combines dense vector search using a fine-tuned SigLIP 2 model with structured attribute filtering to retrieve products from a large-scale catalog.

The dataset used is the Amazon Berkeley Objects (ABO) dataset.

## Project Structure

```
.
├── config.py
├── data_pipeline/
│   ├── download.py
│   ├── schema.py
│   └── clean.py
├── requirements.txt
└── README.md
```

- **config.py**: Holds all paths, model names, and pipeline constants.
- **data_pipeline/download.py**: Verifies the presence of the raw listings and images dataset files on disk.
- **data_pipeline/schema.py**: Contains the Python dataclass definitions representing the clean schema.
- **data_pipeline/clean.py**: Filters, cleans, and deduplicates the raw listings and merges them with the image metadata.

## Phase 1: Data Preparation

The ETL process reads raw listing files and produces a clean, flat Parquet catalog.

### Processing and Filtering Rules
1. **Product Type Filtering**: Excludes all products of type cellular_phone_case.
2. **Text Filtering**: Excludes records where the product name contains the word "swatch" (case-insensitive).
3. **Language Normalization**: Retains English language variants only, falling back sequentially: en_US, en_CA, en_GB, en_AU, en_IN. If no English name variant is found, the record is dropped.
4. **Material Normalization**: Prefers the material description field, falling back to fabric type.
5. **Deduplication**: Deduplicates listings by item_id across marketplaces. When multiple duplicate entries exist, it selects the record corresponding to the country of highest priority (US > CA > GB > AU > IN).

### Output Schema

The output dataset (`products.parquet`) contains exactly the following attributes:
- `item_id`: Unique ASIN identifier.
- `product_type`: Lowercased product type string.
- `title`: First available English product name.
- `color`: Standardized color name or raw color name.
- `material`: Material description or fabric type.
- `brand`: Brand name in English.
- `category`: Leaf segment of the category node.
- `main_image_id`: Unique identifier for the primary product image.
- `main_image_path`: Relative file path to the JPEG image file.

## Getting Started

### Prerequisites

Create a virtual environment and install the package requirements:

```bash
conda create -n ai_env python=3.11
conda activate ai_env
pip install -r requirements.txt
```

### Running Verification & ETL Pipeline

1. Validate that the raw dataset exists in `Raw_Data/`:
   ```bash
   python data_pipeline/download.py
   ```

2. Execute the data preparation and cleaning pipeline:
   ```bash
   python data_pipeline/clean.py
   ```

The cleaned product table and raw images metadata table will be saved to the `Processed_Data/` directory as Parquet files.
