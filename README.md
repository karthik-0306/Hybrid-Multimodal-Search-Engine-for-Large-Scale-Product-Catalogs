---
title: Hybrid Multimodal Search Engine
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Hybrid Multimodal Search Engine for Large-Scale Product Catalogs

A production-ready hybrid search engine that combines dense visual embeddings from a fine-tuned **SigLIP 2** model with sparse keyword retrieval using **SPLADE**, fused via **Reciprocal Rank Fusion (RRF)** and served through a **FastAPI** backend with a premium dark-mode web interface.

The dataset is the Amazon Berkeley Objects (ABO) dataset — **56,427 products** with images, titles, brands, colors, and materials.

---

## 🏗️ Architecture

```
Text Query
    │
    ▼
┌─────────────────────────────────────┐
│         HybridSearchEngine          │
│                                     │
│  ┌───────────────┐  ┌────────────┐  │
│  │  SigLIP 2     │  │  SPLADE    │  │
│  │  Text Tower   │  │  Sparse    │  │
│  │  (768-dim     │  │  Keyword   │  │
│  │   dense vec)  │  │  Vectors   │  │
│  └───────┬───────┘  └─────┬──────┘  │
│          │    RRF Fusion  │         │
│          └────────┬───────┘         │
└───────────────────┼─────────────────┘
                    ▼
            ┌──────────────┐
            │  Qdrant DB   │
            │  56,427 pts  │
            └──────────────┘
                    ▼
            Ranked Results
```

---

## 📦 Pipeline Phases

### Phase 1: Data Preparation & ETL
Reads raw ABO `.json.gz` listing files and produces a clean, flat Parquet catalog.

- **Streaming I/O**: Python generators + `gzip` to process 80GB of raw JSON without crashing RAM
- **Deduplication**: Keeps highest-priority marketplace (US > CA > GB > AU > IN) per ASIN
- **Filtering**: Drops swatch products, unsupported product types, non-English listings, and products with missing images
- **Output Schema**: 9 flat fields per product — `item_id`, `title`, `brand`, `color`, `material`, `product_type`, `category`, `main_image_id`, `main_image_path`

### Phase 2: Multimodal Fine-Tuning (SigLIP 2) — Kaggle GPU
Enhances the baseline model's understanding of our specific product domain.

- **Base Model**: `google/siglip2-base-patch16-224`
- **Technique**: PEFT using **LoRA** (Low-Rank Adaptation, rank=16, alpha=32)
- **Training**: Contrastive image-text learning with compositional captions (`"a brown metal table by Rivet"`)
- **Caption Strategy**: Attribute-only captions (no marketing noise from raw Amazon titles) + brand dropout for robustness
- **Output**: LoRA adapter weights saved to `Processed_Data/models/lora_adapter/`

### Phase 3: Hybrid Vector Indexing (Qdrant) — Kaggle GPU + Local CPU
Builds the searchable vector database using two complementary signals.

- **Dense Embeddings**: Fine-tuned SigLIP 2 image tower → 768-dim vectors per product image (generated on Kaggle GPU, stored as `.npy`)
- **Sparse Embeddings**: `prithivida/Splade_PP_en_v1` via pure PyTorch → sparse keyword importance scores (generated on Kaggle GPU, stored as `.json`)
- **Indexing**: Both vectors + full JSON payload upserted into a local Qdrant collection (`abo_products`)
- **Fusion**: Qdrant's built-in `FusionQuery(RRF)` combines both signals at query time

### Phase 4: Search API & Web Interface
A FastAPI backend exposing the search engine and a premium dark-mode UI.

- **Backend**: FastAPI with lifespan model loading (models loaded once at startup, not per request)
- **Search**: `GET /api/search?q={query}` — runs full hybrid RRF search, returns JSON results
- **Frontend**: Premium dark-mode SPA with glassmorphism, animated product cards, and relevance score bars
- **Image serving**: Static mount of `Processed_Data/images/` at `/images/`

---

## 🚀 Getting Started

### Prerequisites

```bash
conda create -n ai_env python=3.11
conda activate ai_env
pip install -r requirements.txt
```

### Running the Full Pipeline

**Step 1: Data Cleaning**
```bash
python data_pipeline/clean.py
```

**Step 2: Fine-Tuning** *(run on Kaggle GPU — see `data_pipeline/package_for_kaggle.py`)*
```bash
python training/train.py
```

**Step 3: Vector Indexing**
*Automatically uses pre-computed Kaggle vectors from `Processed_Data/` if present.*
```bash
python indexing/build_index.py
```

**Step 4: Launch the Web Interface**
```bash
python app.py
```
Then open **`http://localhost:8000`** in your browser.

---

## 🗂️ Repository Structure

```
├── app.py                        # FastAPI entrypoint — run this to start
├── config.py                     # Centralized paths and constants
├── requirements.txt
│
├── data_pipeline/
│   ├── download.py               # Dataset validation
│   ├── loader.py                 # Streaming gzip JSON reader
│   ├── schema.py                 # Product dataclass + caption builder
│   └── clean.py                  # ETL orchestrator (filter, dedup, export)
│
├── training/
│   ├── lora_config.py            # LoRA hyperparameters
│   └── train.py                  # SigLIP 2 contrastive fine-tuning
│
├── indexing/
│   ├── inference.py              # InferenceEngine (SigLIP 2 + SPLADE)
│   └── build_index.py            # Qdrant collection builder
│
├── search/
│   └── engine.py                 # HybridSearchEngine (text query → RRF results)
│
└── static/
    ├── index.html                # Premium single-page UI
    ├── styles.css                # Dark-mode design system
    └── app.js                    # Search logic, card rendering
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Dense Embeddings | `google/siglip2-base-patch16-224` + LoRA |
| Sparse Embeddings | `prithivida/Splade_PP_en_v1` (SPLADE) |
| Vector Database | Qdrant (local) |
| Fusion Strategy | Reciprocal Rank Fusion (RRF) |
| Backend | FastAPI + Uvicorn |
| Fine-Tuning | HuggingFace PEFT + Transformers |
| Dataset | Amazon Berkeley Objects (ABO) |

---

## 📊 Dataset Stats

| Metric | Value |
|---|---|
| Raw listings parsed | ~147,000 |
| Products after cleaning | 56,427 |
| Dense vector dimensions | 768 |
| Qdrant collection | `abo_products` |
