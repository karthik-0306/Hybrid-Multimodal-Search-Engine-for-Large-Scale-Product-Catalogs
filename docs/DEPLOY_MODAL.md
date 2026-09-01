# Deploying to Modal (free tier)

Modal runs the container only while a request is being served and **scales to
zero** when idle, so the **$30/month free credit** (Starter plan, no credit card
required) easily covers a portfolio demo. The full PyTorch path runs unchanged —
fine-tuned SigLIP 2 + LoRA + SPLADE, no shrinking or re-indexing.

## Prerequisites

- Qdrant Cloud collection `abo_products` populated (via `scripts/cloud_reindex.py`)
- Product images uploaded to a **public** S3 bucket (via `scripts/upload_to_s3.py`)
- Your `QDRANT_URL` and `QDRANT_API_KEY` (in your local `.env`)

## What is in the repo for this

| File | Purpose |
|------|---------|
| `modal_app.py` | Modal image + secret + serves the existing FastAPI `app` as a web endpoint |
| `requirements-modal.txt` | runtime deps (torch installed separately, CPU) |
| `deploy/lora_adapter/` | 4.6 MB LoRA adapter, shipped in the image |

`app.py`, `search/engine.py`, `config.py` are unchanged — with `QDRANT_URL` /
`QDRANT_API_KEY` in the environment they use Qdrant Cloud, and `LORA_ADAPTER_DIR`
(set inside `modal_app.py`) points at the committed adapter.

## Steps

### 1. Sign up
https://modal.com/signup — "Continue with GitHub" or Google. No credit card.

### 2. Install the CLI and log in
```bash
pip install modal
modal token new
```
`modal token new` opens a browser to authorize this machine (one time).

### 3. Create the Qdrant secret
```bash
modal secret create qdrant QDRANT_URL="https://<your-cluster>.qdrant.io" QDRANT_API_KEY="<your-key>"
```
(or in the web UI: Modal dashboard → Secrets → New secret, name it exactly `qdrant`,
add keys `QDRANT_URL` and `QDRANT_API_KEY`.)

### 4. Deploy
```bash
modal deploy modal_app.py
```
The first run builds the image (installs torch + deps, bakes the SigLIP 2 and
SPLADE weights in) — a few minutes. It prints a public URL like:
```
https://<username>--contextclosest-search-web.modal.run
```

### 5. Verify
- `https://<username>--contextclosest-search-web.modal.run/api/health` → `{"status":"ok",...}`
  (first hit after idle is a ~20-40 s cold start while models load into RAM)
- Open the URL in a browser and run a search.

## Redeploying after code changes

```bash
git push origin main        # keep GitHub in sync
modal deploy modal_app.py   # redeploy
```

## Notes / tuning

- **Cost:** you pay only for seconds served. `scaledown_window=300` in `modal_app.py`
  keeps the container warm 5 min after the last request; lower it to save credit,
  raise it to reduce cold starts.
- **Logs:** `modal app logs contextclosest-search` (retained 1 day on Starter).
- **Stop it:** `modal app stop contextclosest-search`.
- If you exceed the $30 credit without adding billing, Modal **stops** the app
  rather than charging you.
