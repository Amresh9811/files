"""
Upload trained models to HuggingFace Hub for deployment.

Usage:
    python upload_to_hub.py --username your-hf-username

Uploads:
  - HuggingFace model  → your-username/multilingual-transliteration-mt5
  - CTranslate2 model  → your-username/multilingual-transliteration-ct2  (if available)
  - Spaces demo        → spaces/your-username/multilingual-transliteration
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--username",      required=True, help="HuggingFace username")
    p.add_argument("--hf_model_dir",  default="checkpoints/best")
    p.add_argument("--ct2_model_dir", default="ct2_model")
    p.add_argument("--demo_dir",      default="demo")
    p.add_argument("--skip_ct2",      action="store_true")
    p.add_argument("--skip_spaces",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    api  = HfApi()

    # ── 1. Upload HF model ────────────────────────────────────────────────
    hf_repo = f"{args.username}/multilingual-transliteration-mt5"
    print(f"Creating model repo: {hf_repo}")
    create_repo(hf_repo, repo_type="model", exist_ok=True)

    print(f"Uploading model from {args.hf_model_dir} …")
    api.upload_folder(
        folder_path=args.hf_model_dir,
        repo_id=hf_repo,
        repo_type="model",
    )
    print(f"✓ HF model uploaded → https://huggingface.co/{hf_repo}")

    # ── 2. Upload CT2 model ───────────────────────────────────────────────
    if not args.skip_ct2 and Path(args.ct2_model_dir).exists():
        ct2_repo = f"{args.username}/multilingual-transliteration-ct2"
        print(f"Creating CT2 model repo: {ct2_repo}")
        create_repo(ct2_repo, repo_type="model", exist_ok=True)

        print(f"Uploading CT2 model from {args.ct2_model_dir} …")
        api.upload_folder(
            folder_path=args.ct2_model_dir,
            repo_id=ct2_repo,
            repo_type="model",
        )
        print(f"✓ CT2 model uploaded → https://huggingface.co/{ct2_repo}")

    # ── 3. Upload Spaces demo ─────────────────────────────────────────────
    if not args.skip_spaces:
        space_repo = f"{args.username}/multilingual-transliteration"
        print(f"Creating Spaces repo: {space_repo}")
        create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)

        # Patch HF_MODEL_ID in the demo before uploading
        app_path = Path(args.demo_dir) / "app.py"
        content  = app_path.read_text()
        content  = content.replace(
            "your-username/multilingual-transliteration-mt5",
            hf_repo,
        )
        app_path.write_text(content)

        print(f"Uploading demo from {args.demo_dir} …")
        api.upload_folder(
            folder_path=args.demo_dir,
            repo_id=space_repo,
            repo_type="space",
        )
        print(f"✓ Spaces demo uploaded → https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
