# Hugging Face Spaces (Docker SDK) — free CPU tier: 2 vCPU / 16 GB RAM.
# Runs the full PyTorch inference path (fine-tuned SigLIP 2 + LoRA + SPLADE),
# so no ONNX / quantization is needed here.

FROM python:3.11-slim

# Spaces runs the container as uid 1000; give that user a home for caches.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    PYTHONUNBUFFERED=1 \
    LORA_ADAPTER_DIR=/app/deploy/lora_adapter

WORKDIR /app

# CPU-only torch first (avoids pulling ~2 GB of CUDA wheels)
COPY --chown=user requirements-hf.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-hf.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
