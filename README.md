# Hybrid Multimodal Search Engine for Large Scale Product Catalogs

This repository contains a production-ready hybrid search engine that combines dense vector search using a fine-tuned **SigLIP 2** model with sparse keyword retrieval using **SPLADE**, all stored efficiently in **Qdrant** to retrieve products from a large-scale catalog.

The dataset used is the Amazon Berkeley Objects (ABO) dataset, comprising over 56,000 processed products and images.

## 🏗️ Architecture & Pipeline

### Phase 1: Data Preparation & Cleaning
The ETL process reads raw listing files and produces a clean, flat Parquet catalog (`products.parquet`).
- **Deduplication**: Removes exact duplicates and standardizes attributes across multiple international marketplaces.
- **Filtering**: Specifically isolates relevant product categories (excluding items like phone cases or watches) and normalizes languages to English.
- **Output Schema**: Extracts exact text payloads (`title`, `brand`, `color`, `material`, `category`) and maps them to their respective visual images.

### Phase 2: Multimodal Fine-Tuning (SigLIP 2)
To enhance the baseline model's understanding of our specific product catalog:
- **Base Model**: `google/siglip2-base-patch16-224`
- **Technique**: Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Low-Rank Adaptation).
- **Process**: Fine-tuned the Vision and Text encoders simultaneously on AWS/Kaggle infrastructure to project domain-specific vocabulary (brand names, specific colors, materials) into a joint embedding space.
- **Output**: LoRA adapter weights saved in `Processed_Data/models/lora_adapter`.

### Phase 3: Hybrid Vector Indexing (Qdrant)
To provide extremely accurate search results, we rely on Reciprocal Rank Fusion (RRF) to merge two separate embedding models:
- **Dense Visual Embeddings**: Uses the fine-tuned SigLIP 2 model to extract 768-dimensional vectors from product images.
- **Sparse Text Embeddings**: Uses `prithivida/Splade_PP_en_v1` via PyTorch/FastEmbed to extract keyword importance scores from the product metadata.
- **Database**: Both vectors and the JSON payload are indexed locally using `Qdrant`, providing sub-second hybrid retrieval times.

---

## 🚀 Getting Started

### Prerequisites

Create a virtual environment and install the package requirements:

```bash
conda create -n ai_env python=3.11
conda activate ai_env
pip install -r requirements.txt
```

### Running the Pipeline

**1. Data Preparation**
```bash
python data_pipeline/clean.py
```

**2. Model Fine-Tuning**
```bash
python training/train.py
```

**3. Vector Indexing**
*Note: This process leverages pre-computed Kaggle vectors if placed in `Processed_Data/`, or automatically falls back to local CPU processing if missing.*
```bash
python indexing/build_index.py
```

**4. Testing the Search Engine**
```bash
python test_search.py
```
