"""
indexing/inference.py

Handles loading the fine-tuned SigLIP 2 model and the sparse SPLADE model.
Supports dual-mode: PyTorch (local) and ONNX (Render cloud deployment).
"""

import os
import logging
from pathlib import Path
import numpy as np

# fastembed natively uses ONNX under the hood and is lightweight
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
        """
        self.is_render = os.getenv("RENDER") == "True"
        
        # 1. Load SigLIP 2 Dense Model
        if self.is_render:
            logger.info("Initializing InferenceEngine in RENDER mode (ONNX).")
            import onnxruntime as ort
            from transformers import AutoProcessor
            
            onnx_path = Path("models/onnx/siglip_text.onnx")
            if not onnx_path.exists():
                logger.error(f"ONNX model not found at {onnx_path}!")
                
            self.processor = AutoProcessor.from_pretrained(base_model_name)
            self.ort_session = ort.InferenceSession(str(onnx_path))
            
        else:
            logger.info("Initializing InferenceEngine in LOCAL mode (PyTorch).")
            import torch
            from peft import PeftModel
            from transformers import AutoModel, AutoProcessor
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def embed_text_dense(self, texts: list[str]) -> list[list[float]]:
        """
        Generates dense SigLIP 2 embeddings for text queries.
        """
        if self.is_render:
            # ONNX Inference
            inputs = self.processor(
                text=texts,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            ort_inputs = {"input_ids": inputs["input_ids"]}
            
            # pooler_output is returned
            pooler_output = self.ort_session.run(None, ort_inputs)[0]
            
            # L2 Normalize
            norms = np.linalg.norm(pooler_output, axis=1, keepdims=True)
            normalized = pooler_output / norms
            return normalized.tolist()
            
        else:
            # PyTorch Inference
            import torch
            inputs = self.processor(
                text=texts,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                text_embeddings = self.dense_model.base_model.get_text_features(**inputs)
                
                if hasattr(text_embeddings, "text_embeds"):
                    text_embeddings = text_embeddings.text_embeds
                elif hasattr(text_embeddings, "pooler_output"):
                    text_embeddings = text_embeddings.pooler_output
                elif not isinstance(text_embeddings, torch.Tensor):
                    text_embeddings = text_embeddings[0]

                text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
                return text_embeddings.cpu().numpy().tolist()

    def embed_text_sparse(self, texts: list[str]) -> list[dict]:
        """
        Generates sparse SPLADE embeddings for text.
        """
        embeddings_iter = self.sparse_model.embed(texts)
        results = []
        for emb in embeddings_iter:
            results.append({
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist()
            })
        return results

    def embed_images_dense(self, images: list) -> list[list[float]]:
        """
        Generates dense SigLIP 2 embeddings for images locally.
        Not supported in Render mode.
        """
        if self.is_render:
            raise NotImplementedError("Image embedding is not supported on Render.")
            
        import torch
        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            image_embeddings = self.dense_model.base_model.get_image_features(pixel_values=inputs["pixel_values"])
            
            if hasattr(image_embeddings, "image_embeds"):
                image_embeddings = image_embeddings.image_embeds
            elif hasattr(image_embeddings, "pooler_output"):
                image_embeddings = image_embeddings.pooler_output
            elif not isinstance(image_embeddings, torch.Tensor):
                image_embeddings = image_embeddings[0]

            image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
            return image_embeddings.cpu().numpy().tolist()
