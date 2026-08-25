"""
indexing/build_index.py

Builds the Qdrant hybrid vector index for the ABO product catalog.
It loads pre-computed image dense vectors (from Kaggle) or generates them locally,
computes sparse text vectors for keyword search, and uploads everything to Qdrant.
"""

import logging
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct, SparseVector
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from indexing.inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "abo_products"
BATCH_SIZE = 256


def initialize_qdrant(config: Config) -> QdrantClient:
    """
    Initializes a local Qdrant database and configures the hybrid collection.
    """
    config.qdrant_db_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(config.qdrant_db_dir))

    # Recreate the collection to ensure a fresh start
    if client.collection_exists(COLLECTION_NAME):
        logger.info("Dropping existing collection '%s'", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    logger.info("Creating hybrid collection '%s'", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense_image": VectorParams(size=768, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse_text": SparseVectorParams(),
        }
    )
    return client


def build_search_text(record: dict) -> str:
    """
    Combines fields into a single rich text document for sparse embedding (keyword search).
    """
    parts = [
        str(record.get('title', '')),
        str(record.get('brand', '')),
        str(record.get('color', '')),
        str(record.get('material', '')),
        str(record.get('category', ''))
    ]
    return " ".join([p for p in parts if p and p != "None"])


def main() -> None:
    config = Config(Path(__file__).resolve().parent.parent)
    
    logger.info("Loading products from %s", config.products_parquet)
    df = pd.read_parquet(config.products_parquet)
    records = df.to_dict(orient="records")
    logger.info("Total products to index: %d", len(records))

    # 1. Load or Generate Dense Image Vectors
    if config.image_embeddings_npy.exists():
        logger.info("Found pre-computed image embeddings from Kaggle at %s", config.image_embeddings_npy)
        dense_vectors = np.load(config.image_embeddings_npy)
        if len(dense_vectors) != len(records):
            raise ValueError(f"Mismatch! Found {len(dense_vectors)} embeddings but {len(records)} products.")
        engine = InferenceEngine(config.BASE_MODEL_NAME, str(config.lora_adapter_dir))
    else:
        logger.warning("No pre-computed Kaggle embeddings found!")
        logger.warning("Generating dense image embeddings locally (THIS WILL BE SLOW ON CPU)...")
        from PIL import Image
        engine = InferenceEngine(config.BASE_MODEL_NAME, str(config.lora_adapter_dir))
        
        dense_vectors = []
        for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Extracting Dense Vectors"):
            batch = records[i:i+BATCH_SIZE]
            images = []
            for r in batch:
                img_path = config.images_base_dir / r['main_image_path']
                try:
                    images.append(Image.open(img_path).convert('RGB'))
                except Exception:
                    images.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))
            embs = engine.embed_images_dense(images)
            dense_vectors.extend(embs)
        dense_vectors = np.array(dense_vectors)

    # 2. Generate or Load Sparse Text Vectors
    sparse_embeddings_json = config.processed_dir / "sparse_text_embeddings.json"
    if sparse_embeddings_json.exists():
        logger.info("Found pre-computed sparse text embeddings from Kaggle!")
        import json
        with open(sparse_embeddings_json, 'r') as f:
            sparse_vectors = json.load(f)
    else:
        logger.warning("No pre-computed text embeddings found. Generating locally (this may be slow)...")
        logger.info("Generating Sparse (SPLADE) Text Vectors...")
        search_texts = [build_search_text(r) for r in records]
        
        sparse_vectors = []
        for i in tqdm(range(0, len(search_texts), BATCH_SIZE), desc="Extracting Sparse Vectors"):
            batch = search_texts[i:i+BATCH_SIZE]
            sparse_vectors.extend(engine.embed_text_sparse(batch))

    # 3. Insert into Qdrant
    client = initialize_qdrant(config)
    
    points = []
    for idx, record in enumerate(records):
        # We generate a deterministic UUID based on the ASIN (item_id)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(record["item_id"])))
        
        points.append(
            PointStruct(
                id=point_id,
                payload=record,  # Store the entire cleaned product row
                vector={
                    "dense_image": dense_vectors[idx].tolist(),
                    "sparse_text": SparseVector(
                    indices=sparse_vectors[idx]["indices"] if isinstance(sparse_vectors[idx], dict) else sparse_vectors[idx].indices.tolist(),
                    values=sparse_vectors[idx]["values"] if isinstance(sparse_vectors[idx], dict) else sparse_vectors[idx].values.tolist()
                )
                }
            )
        )

    logger.info("Uploading vectors to Qdrant in batches...")
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Uploading to Qdrant"):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i+BATCH_SIZE]
        )

    logger.info("Index build complete! Vectors are safely stored in Qdrant.")


if __name__ == "__main__":
    main()
