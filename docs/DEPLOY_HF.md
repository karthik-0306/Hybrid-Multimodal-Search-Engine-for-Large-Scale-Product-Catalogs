# Deploying to Hugging Face Spaces (free CPU tier)

The Render free tier (512 MB RAM) cannot hold the SigLIP 2 text encoder
(256k-token Gemma vocab, ~1.1 GB fp32) alongside the SPLADE model. Hugging Face
Spaces' free CPU tier gives **2 vCPU / 16 GB RAM / 50 GB disk**, which runs the
full fine-tuned PyTorch path with no ONNX or quantization.

## What is already in the repo for this

| File | Purpose |
|------|---------|
| `Dockerfile` | Docker SDK Space; installs CPU torch + `requirements-hf.txt`; serves on port 7860 |
| `requirements-hf.txt` | Serving deps (torch installed separately in the Dockerfile) |
| `deploy/lora_adapter/` | 4.7 MB LoRA adapter (config + safetensors) so the Space needs no `Processed_Data/` |
| `README.md` front matter | `sdk: docker`, `app_port: 7860` — the Space config block |
| `config.py` | Picks up `LORA_ADAPTER_DIR` env var (set in the Dockerfile) |

The search engine already talks to **Qdrant Cloud** (via `QDRANT_URL` /
`QDRANT_API_KEY` env vars) and renders images from the **public S3 URLs** stored
in each Qdrant payload — no code change needed there.

## One-time setup

1. Create the Space: https://huggingface.co/new-space
   - **SDK:** Docker → *Blank*
   - **Hardware:** CPU basic (free)
   - Name it e.g. `hybrid-multimodal-search`

2. Add secrets in **Space → Settings → Variables and secrets**:
   - `QDRANT_URL` = your Qdrant Cloud cluster URL
   - `QDRANT_API_KEY` = your Qdrant Cloud API key

3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/hybrid-multimodal-search
   git push space main
   ```
   (Use an HF access token with *write* scope when prompted for a password.)

4. Watch the build logs in the Space UI. First boot downloads the SigLIP 2 base
   weights + SPLADE model (~2 GB, cached afterwards), so it takes a few minutes.

## Verifying

- `https://<your-username>-hybrid-multimodal-search.hf.space/api/health` → `{"status":"ok",...}`
- Open the Space URL and run a search.

## Notes

- Free Spaces sleep after 48 h of inactivity and cold-start on the next request.
- Keep the Space **public** (private Spaces need a paid plan).
- `render.yaml` and `requirements-render.txt` are left in place only for a future
  paid-instance (≥2 GB) Render deployment; they are not used by HF Spaces.
