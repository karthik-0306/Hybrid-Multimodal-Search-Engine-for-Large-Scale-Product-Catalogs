"""
search/engine.py

Provides the HybridSearchEngine class that orchestrates text query embedding
and hybrid vector search against the local Qdrant database.
"""

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
    against a local Qdrant collection using Reciprocal Rank Fusion (RRF).

    The dense vector is produced by the fine-tuned SigLIP 2 text tower.
    The sparse vector is produced by SPLADE for keyword-level matching.
    Both signals are combined via Qdrant's built-in RRF fusion.
    """

    def __init__(self, config: Any) -> None:
        """
        Initializes the search engine by loading the inference models and
        connecting to the local Qdrant database.

        Args:
            config: Project Config instance with paths and settings.
        """
        from indexing.inference import InferenceEngine

        self._config = config

        logger.info("Loading InferenceEngine (SigLIP 2 + SPLADE)...")
        self._inference = InferenceEngine(
            base_model_name=config.BASE_MODEL_NAME,
            lora_adapter_path=str(config.lora_adapter_dir),
        )

        logger.info("Connecting to local Qdrant at %s", config.qdrant_db_dir)
        self._client = QdrantClient(path=str(config.qdrant_db_dir))

        # Validate collection exists
        if not self._client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' not found. "
                "Please run indexing/build_index.py first."
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

        Generates a dense embedding (SigLIP 2 text tower) and a sparse
        embedding (SPLADE) from the query, then uses Qdrant's Reciprocal
        Rank Fusion to combine both signals into a single ranked result list.

        Args:
            query: The natural language search query string.
            top_k: Number of results to return (default 10).

        Returns:
            A dict with keys:
                - "results": list of product dicts (score + all payload fields)
                - "elapsed_ms": float, query latency in milliseconds
                - "count": int, number of results returned
        """
        if not query or not query.strip():
            return {"results": [], "elapsed_ms": 0.0, "count": 0}

        query = query.strip()
        t0 = time.perf_counter()

        # 1. Generate dense text embedding (768-dim, L2-normalized)
        logger.debug("Embedding query (dense): '%s'", query)
        dense_vector: list[float] = self._inference.embed_text_dense([query])[0]

        # 2. Generate sparse SPLADE embedding (keyword indices + weights)
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
                # Dense leg: retrieve top-k candidates by image similarity
                Prefetch(
                    query=dense_vector,
                    using="dense_image",
                    limit=top_k * 3,
                ),
                # Sparse leg: retrieve top-k candidates by keyword match
                Prefetch(
                    query=sparse_vector,
                    using="sparse_text",
                    limit=top_k * 3,
                ),
            ],
            # RRF re-ranks the merged candidates from both legs
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 4. Format results
        formatted = []
        for point in results.points:
            payload = point.payload or {}
            image_path = payload.get("main_image_path", "")
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
                    # Build the URL path for our static image server endpoint
                    "image_url": f"/images/{image_path}" if image_path else "",
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
