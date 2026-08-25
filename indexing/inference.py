"""
indexing/inference.py

Handles loading the fine-tuned SigLIP 2 model and the sparse SPLADE model.
This module is designed to run locally on CPU for fast inference of text queries.
"""

import logging
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModel, AutoProcessor
from fastembed import SparseTextEmbedding

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Wraps the fine-tuned SigLIP 2 model (dense) and SPLADE model (sparse)
    for embedding generation.
    """

    def __init__(self, base_model_name: str, lora_adapter_path: str) -> None:
        """
        Initializes the dense and sparse embedding models.

        Args:
            base_model_name: The HuggingFace ID of the base SigLIP 2 model.
            lora_adapter_path: Path to the locally saved LoRA adapter weights.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing InferenceEngine on %s", self.device)

        # 1. Load SigLIP 2 Dense Model
        logger.info("Loading base model: %s", base_model_name)
        self.processor = AutoProcessor.from_pretrained(base_model_name)
        base_model = AutoModel.from_pretrained(base_model_name)

        if Path(lora_adapter_path).exists():
            logger.info("Applying LoRA adapter from %s", lora_adapter_path)
            self.dense_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        else:
            logger.warning("LoRA adapter not found. Using base model.")
            self.dense_model = base_model

        self.dense_model.to(self.device)
        self.dense_model.eval()

        # 2. Load SPLADE Sparse Model (FastEmbed)
        logger.info("Loading SPLADE sparse embedding model...")
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    @torch.no_grad()
    def embed_text_dense(self, texts: list[str]) -> list[list[float]]:
        """
        Generates dense SigLIP 2 embeddings for text queries.
        """
        inputs = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        text_embeddings = self.dense_model.base_model.get_text_features(**inputs)
        
        if hasattr(text_embeddings, "text_embeds"):
            text_embeddings = text_embeddings.text_embeds
        elif hasattr(text_embeddings, "pooler_output"):
            text_embeddings = text_embeddings.pooler_output
        elif not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = text_embeddings[0]

        # Normalize
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
        return text_embeddings.cpu().numpy().tolist()

    def embed_text_sparse(self, texts: list[str]) -> list[dict]:
        """
        Generates sparse SPLADE embeddings for text.
        Returns a list of dicts, e.g. [{"indices": [...], "values": [...]}, ...]
        """
        embeddings_iter = self.sparse_model.embed(texts)
        results = []
        for emb in embeddings_iter:
            results.append({
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist()
            })
        return results

    @torch.no_grad()
    def embed_images_dense(self, images: list) -> list[list[float]]:
        """
        Fallback method to generate dense SigLIP 2 embeddings for images locally.
        (Primarily used if not using pre-computed Kaggle vectors).
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        image_embeddings = self.dense_model.base_model.get_image_features(pixel_values=inputs["pixel_values"])
        
        # Depending on the specific HF SigLIP wrapper used, this might return an object instead of a tensor
        if hasattr(image_embeddings, "image_embeds"):
            image_embeddings = image_embeddings.image_embeds
        elif hasattr(image_embeddings, "pooler_output"):
            image_embeddings = image_embeddings.pooler_output
        elif not isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings[0]

        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        return image_embeddings.cpu().numpy().tolist()
