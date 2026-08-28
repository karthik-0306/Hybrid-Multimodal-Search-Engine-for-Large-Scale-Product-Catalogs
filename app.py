"""
app.py

FastAPI web application for the Hybrid Multimodal Search Engine.

Run locally with:
    conda activate ai_env
    python app.py

Then open: http://localhost:8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from search.engine import HybridSearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — models are loaded once at startup, not per request
# ---------------------------------------------------------------------------
_engine: HybridSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads the search engine (models + Qdrant connection) once at startup,
    and cleans up on shutdown.
    """
    global _engine
    logger.info("=== Starting Hybrid Search Engine ===")
    config = Config(Path(__file__).resolve().parent)
    _engine = HybridSearchEngine(config)
    logger.info("=== Search Engine Ready — visit http://localhost:8000 ===")
    yield
    logger.info("=== Shutting down ===")
    _engine = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hybrid Multimodal Search Engine",
    description="Text-based hybrid search over 56,000+ Amazon product listings.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve product images from Processed_Data/images/
_project_root = Path(__file__).resolve().parent
_images_dir = _project_root / "Processed_Data" / "images"

if _images_dir.exists():
    app.mount("/images", StaticFiles(directory=str(_images_dir)), name="images")
    logger.info("Mounted product images from %s", _images_dir)
else:
    logger.warning("Images directory not found at %s — images will not display.", _images_dir)

# Serve frontend static files (CSS, JS)
_static_dir = _project_root / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serves the main frontend HTML page."""
    index_path = _static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index_path))


@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=1, description="Text search query"),
    top_k: int = Query(default=12, ge=1, le=50, description="Number of results"),
) -> JSONResponse:
    """
    Runs a hybrid text search (dense SigLIP 2 + sparse SPLADE) against the
    Qdrant collection and returns ranked product results.

    Args:
        q: The natural language query string.
        top_k: Number of results to return (1–50, default 12).

    Returns:
        JSON object with keys: results (list), count (int), elapsed_ms (float).
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready yet.")

    try:
        response = _engine.search(query=q, top_k=top_k)
        return JSONResponse(content=response)
    except Exception as exc:
        logger.exception("Search error for query '%s': %s", q, exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(exc)}")


@app.get("/api/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "ok" if _engine is not None else "loading",
            "collection": "abo_products",
        }
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Reload disabled — models take time to load
        log_level="info",
    )
