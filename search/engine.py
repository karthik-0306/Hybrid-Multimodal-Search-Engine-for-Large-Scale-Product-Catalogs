"""
search/engine.py

Provides the HybridSearchEngine class that orchestrates text query embedding
and hybrid vector search against the Qdrant database (Local or Cloud).
"""

import os
import logging
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "abo_products"


class HybridSearchEngine:
    """
    Orchestrates text-query embedding and hybrid (dense + sparse) search
    against a Qdrant collection using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, config: Any) -> None:
        """
        Initializes the search engine by loading the inference models and
        connecting to the Qdrant database.
        """
        from indexing.inference import InferenceEngine

        self._config = config

        logger.info("Loading InferenceEngine (SigLIP 2 + SPLADE)...")
        self._inference = InferenceEngine(
            base_model_name=config.BASE_MODEL_NAME,
            lora_adapter_path=str(config.lora_adapter_dir),
        )

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            logger.info("Connecting to Qdrant Cloud at %s", qdrant_url)
            self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            logger.info("Connecting to local Qdrant at %s", config.qdrant_db_dir)
            self._client = QdrantClient(path=str(config.qdrant_db_dir))

        # Validate collection exists
        if not self._client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' not found."
            )

        info = self._client.get_collection(COLLECTION_NAME)
        logger.info(
            "Connected to collection '%s' with %d points.",
            COLLECTION_NAME,
            info.points_count,
        )

    def search(self, query: str, top_k: int = 10) -> dict:
        """
        Runs a hybrid text search against the Qdrant collection.
        """
        if not query or not query.strip():
            return {"results": [], "elapsed_ms": 0.0, "count": 0}

        query = query.strip()
        t0 = time.perf_counter()

        # 1. Generate dense text embedding
        logger.debug("Embedding query (dense): '%s'", query)
        dense_vector: list[float] = self._inference.embed_text_dense([query])[0]

        # 2. Generate sparse SPLADE embedding
        logger.debug("Embedding query (sparse): '%s'", query)
        sparse_raw: dict = self._inference.embed_text_sparse([query])[0]
        sparse_vector = SparseVector(
            indices=sparse_raw["indices"],
            values=sparse_raw["values"],
        )

        # 3. Hybrid search with RRF via Qdrant Prefetch + FusionQuery
        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense_image",
                    limit=top_k * 3,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse_text",
                    limit=top_k * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 4. Format results
        formatted = []
        for point in results.points:
            payload = point.payload or {}
            
            # Use public S3 URL if on cloud, otherwise fallback to local path
            if "image_url" in payload:
                image_url = payload["image_url"]
            else:
                image_path = payload.get("main_image_path", "")
                image_url = f"/images/{image_path}" if image_path else ""

            formatted.append(
                {
                    "score": round(point.score, 4),
                    "item_id": payload.get("item_id", ""),
                    "title": payload.get("title", ""),
                    "brand": payload.get("brand") or "",
                    "color": payload.get("color") or "",
                    "material": payload.get("material") or "",
                    "product_type": payload.get("product_type") or "",
                    "category": payload.get("category") or "",
                    "image_url": image_url,
                }
            )

        logger.info(
            "Query '%s' → %d results in %.1f ms", query, len(formatted), elapsed_ms
        )

        return {
            "results": formatted,
            "elapsed_ms": round(elapsed_ms, 1),
            "count": len(formatted),
        }
