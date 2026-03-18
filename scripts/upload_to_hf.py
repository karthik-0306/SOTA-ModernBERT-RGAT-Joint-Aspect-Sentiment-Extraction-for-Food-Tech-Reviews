"""
Upload model checkpoints to Hugging Face Hub.
================================================
One-time script to push large checkpoint files to HF Hub,
so they can be downloaded at deployment time.

Usage:
    python scripts/upload_to_hf.py --token hf_YOUR_TOKEN

This creates/updates the repo: karthik0306/ModernBERT-RGAT-ABSA
"""

import os
import argparse
from pathlib import Path


def upload_checkpoints(token: str, repo_id: str = "karthik0306/ModernBERT-RGAT-ABSA"):
    """Upload all checkpoint files to HF Hub."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"[OK] Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[WARN] Repo creation note: {e}")

    # Find checkpoint files
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / "checkpoints"

    if not checkpoint_dir.exists():
        print(f"[ERROR] Checkpoint directory not found: {checkpoint_dir}")
        return

    files = sorted(checkpoint_dir.glob("best_model_*.pt"))
    if not files:
        print("[ERROR] No checkpoint files found!")
        return

    print(f"\nFound {len(files)} checkpoint(s) to upload:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {f.name}  ({size_mb:.0f} MB)")

    # Upload each file
    for f in files:
        print(f"\n[UPLOAD] {f.name}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"   [OK] {f.name} uploaded!")

    # Also upload config for reproducibility
    config_path = project_root / "configs" / "config.yaml"
    if config_path.exists():
        print(f"\n[UPLOAD] config.yaml...")
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.yaml",
            repo_id=repo_id,
            repo_type="model",
        )
        print("   [OK] config.yaml uploaded!")

    print(f"\n[DONE] All files uploaded to: https://huggingface.co/{repo_id}")
    print(f"   Set HF_MODEL_REPO={repo_id} in your deployment environment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload checkpoints to HF Hub")
    parser.add_argument("--token", required=True, help="HF access token (write)")
    parser.add_argument(
        "--repo",
        default="karthik0306/ModernBERT-RGAT-ABSA",
        help="HF repo ID (default: karthik0306/ModernBERT-RGAT-ABSA)",
    )
    args = parser.parse_args()
    upload_checkpoints(token=args.token, repo_id=args.repo)
