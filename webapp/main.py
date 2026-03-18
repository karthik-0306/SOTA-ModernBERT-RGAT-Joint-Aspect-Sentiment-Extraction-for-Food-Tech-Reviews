"""
ModernBERT-RGAT | FastAPI Web Application
============================================
REST API backend for Aspect-Based Sentiment Analysis.

Endpoints:
    GET  /            → Serves the frontend
    GET  /api/health  → Model status
    POST /api/analyze → Run inference on a review

Run locally:
    uvicorn webapp.main:app --host 0.0.0.0 --port 7860 --reload
"""

import os
import sys
import time
from typing import Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.schemas import AnalyzeRequest, AnalyzeResponse, AspectResult, HealthResponse
from src.inference import AspectSentimentPredictor, load_predictor


# ─── Global State ─────────────────────────────────────────────────

_predictors: Dict[str, AspectSentimentPredictor] = {}
_device: Optional[torch.device] = None
_best_year: Optional[str] = None


def _load_models():
    """Load the best available model checkpoint at startup."""
    global _predictors, _device, _best_year

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {_device}")

    # Try to download from HF Hub if checkpoints don't exist locally
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    _try_download_from_hub(checkpoint_dir)

    # Load best available model (most recent year first)
    for year in ["2016", "2015", "2014"]:
        ckpt_path = os.path.join(checkpoint_dir, f"best_model_{year}.pt")
        if os.path.exists(ckpt_path):
            try:
                print(f"  Loading {year} model from {ckpt_path}...")
                _predictors[year] = load_predictor(
                    checkpoint_path=ckpt_path, device=_device
                )
                _best_year = year
                print(f"  [OK] {year} model loaded successfully")
                break  # Only load one model to save memory on free tier
            except Exception as e:
                import traceback
                print(f"  [ERROR] Failed to load {year}: {e}")
                traceback.print_exc()

    if not _predictors:
        print("[WARN] No models loaded! Inference will not work.")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Exists: {os.path.exists(checkpoint_dir)}")
        if os.path.exists(checkpoint_dir):
            print(f"  Contents: {os.listdir(checkpoint_dir)}")
    else:
        print(f"[INFO] {len(_predictors)} model(s) ready. Best: {_best_year}")


def _try_download_from_hub(checkpoint_dir: str):
    """Download checkpoints from Hugging Face Hub if not present locally."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if any checkpoint already exists
    existing = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("best_model_") and f.endswith(".pt")
    ]
    if existing:
        print(f"  [INFO] Found local checkpoints: {existing}")
        return  # Already have local checkpoints

    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        print("  [WARN] No HF_MODEL_REPO set, skipping HF Hub download.")
        return

    print(f"  [INFO] Downloading models from HF Hub: {repo_id}")
    try:
        from huggingface_hub import hf_hub_download
        # Only download the best model (2016) to save time and disk
        for year in ["2016", "2015", "2014"]:
            filename = f"best_model_{year}.pt"
            print(f"  [DOWNLOAD] {filename} from {repo_id}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=checkpoint_dir,
                token=os.environ.get("HF_TOKEN"),
            )
            print(f"  [OK] Downloaded {filename} to {downloaded_path}")
            break  # Only need one model
    except Exception as e:
        import traceback
        print(f"  [WARN] HF Hub download failed: {e}")
        traceback.print_exc()


# ─── FastAPI App ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    _load_models()
    yield
    # Cleanup on shutdown (if needed)
    _predictors.clear()


app = FastAPI(
    title="ModernBERT-RGAT | ABSA API",
    description="Joint Aspect Extraction & Sentiment Classification for Restaurant Reviews",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files (frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Routes ───────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the single-page frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Use /docs for the API."}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health and loaded models."""
    return HealthResponse(
        status="healthy" if _predictors else "no_models",
        models_loaded=list(_predictors.keys()),
        device=str(_device) if _device else "unknown",
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_review(request: AnalyzeRequest):
    """
    Analyze a restaurant review for aspects and sentiments.

    - **text**: The review text to analyze (1–2000 characters)
    - **model_year**: Which model to use: "best", "2014", "2015", or "2016"
    """
    if not _predictors:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Please check server logs.",
        )

    # Resolve model year
    year = request.model_year.strip().lower()
    if year == "best":
        year = _best_year
    if year not in _predictors:
        available = ", ".join(sorted(_predictors.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_year}' not available. Loaded: {available}",
        )

    predictor = _predictors[year]

    # Run inference with timing
    start_time = time.perf_counter()
    try:
        predictions = predictor.predict(request.text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Convert to response
    aspects = [
        AspectResult(
            aspect=p.aspect,
            sentiment=p.sentiment,
            confidence=round(p.confidence, 4),
            start=p.start,
            end=p.end,
        )
        for p in predictions
    ]

    return AnalyzeResponse(
        text=request.text.strip(),
        model_used=year,
        aspects=aspects,
        processing_time_ms=round(elapsed_ms, 1),
    )


# ─── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "webapp.main:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
    )
