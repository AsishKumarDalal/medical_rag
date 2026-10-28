"""Small OpenAI-compatible embedding server for a local sentence-transformers model.

Run:
    .venv/bin/python data_test/embedding_server.py --model data_test/embeddings/bge-small-en-v1.5 --port 8081

Exposes:
    POST /v1/embeddings  {"model": "...", "input": ["text", ...]}
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}. Install fastapi and uvicorn.") from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}. Install sentence-transformers.") from exc


class EmbedRequest(BaseModel):
    model: str | None = None
    input: Any


def create_app(model_path: str) -> FastAPI:
    app = FastAPI()
    model = SentenceTransformer(model_path)

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_path, "object": "model"}]}

    @app.post("/v1/embeddings")
    def embeddings(req: EmbedRequest) -> dict[str, Any]:
        texts = req.input if isinstance(req.input, list) else [req.input]
        vectors = model.encode(texts, normalize_embeddings=True)
        data = [
            {"object": "embedding", "index": i, "embedding": [float(x) for x in vec]}
            for i, vec in enumerate(vectors)
        ]
        return {"object": "list", "data": data, "model": req.model or model_path}

    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model path not found: {args.model}")

    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
