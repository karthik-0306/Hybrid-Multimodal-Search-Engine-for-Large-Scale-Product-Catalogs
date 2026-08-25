import logging
from config import Config
from pathlib import Path
from indexing.inference import InferenceEngine
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, QueryRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_search():
    config = Config(Path('.'))
    
    logger.info("Initializing Inference Engine...")
    engine = InferenceEngine(config.BASE_MODEL_NAME, str(config.lora_adapter_dir))
    
    logger.info("Connecting to Qdrant...")
    client = QdrantClient(path=str(config.qdrant_db_dir))
    
    query = "stylish black leather boots"
    logger.info(f"Test Query: '{query}'")
    
    logger.info("Extracting embeddings...")
    dense_query = engine.embed_text_dense([query])[0]
    sparse_query = engine.embed_text_sparse([query])[0]
    
    logger.info("Executing Qdrant Query...")
    # Hybrid search using Reciprocal Rank Fusion (RRF) in Qdrant
    results = client.query_points(
        collection_name="abo_products",
        prefetch=[
            Prefetch(query=dense_query, using="dense_image", limit=10),
            Prefetch(
                query=qdrant_client.models.SparseVector(
                    indices=sparse_query["indices"],
                    values=sparse_query["values"]
                ),
                using="sparse_text",
                limit=10
            )
        ],
        query=qdrant_client.models.FusionQuery(fusion=qdrant_client.models.Fusion.RRF),
        limit=3,
        with_payload=["title", "brand"]
    )
    
    logger.info("Search Results:")
    for i, res in enumerate(results.points):
        logger.info(f"[{i+1}] Score: {res.score:.4f} | Title: {res.payload.get('title')}")

if __name__ == "__main__":
    import qdrant_client
    test_hybrid_search()
