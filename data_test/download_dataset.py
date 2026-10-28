"""Download the public Hugging Face dataset bundle into data_test/.

Usage:
    HF_REPO_ID=yourname/medical-graphrag-data-test HF_TOKEN=hf_xxx \
        .venv/bin/python data_test/download_dataset.py

The dataset is expected to contain:
    corpus/   artifacts/   models/   embeddings/   plan_data.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_TEST = Path(__file__).resolve().parent


def main() -> int:
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    if not repo_id:
        print("Set HF_REPO_ID, e.g. username/medical-graphrag-data-test", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN", "").strip() or None

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed; installing...", file=sys.stderr)
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    print(f"Downloading dataset {repo_id} into {DATA_TEST} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(DATA_TEST),
        token=token,
        resume_download=True,
    )
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
