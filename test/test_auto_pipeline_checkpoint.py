"""Unit tests for crash-safe HF checkpoints in data_test/auto_pipeline.py."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from data_test import auto_pipeline


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_progress_tracks_stages(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    write_json(artifacts / "index.json", {"article_count": 3, "chunk_count": 45})
    (artifacts / "triples.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"chunk_id": "a::1", "triples": [{"head": "A"}, {"head": "B"}]}),
                json.dumps({"chunk_id": "a::2", "triples": [{"head": "C"}]}),
                "{partial",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(artifacts / "failed_chunks.json", {"count": 1, "chunks": [{"chunk_id": "a::3"}]})

    progress = auto_pipeline.collect_progress(tmp_path, started_at=0)
    assert progress["stage"] == "extract"
    assert progress["chunks_total"] == 45
    assert progress["chunks_extracted"] == 2
    assert progress["chunks_failed"] == 1
    assert progress["chunks_done"] == 3
    assert progress["triples"] == 3
    assert progress["percent"] == 6.7

    write_json(artifacts / "graph.json", {"nodes": {"a": {}, "b": {}}, "edges": [{}]})
    progress = auto_pipeline.collect_progress(tmp_path)
    assert progress["stage"] == "graph"
    assert progress["nodes"] == 2
    assert progress["edges"] == 1

    (artifacts / "merge_log.json").write_text("{}", encoding="utf-8")
    assert auto_pipeline.collect_progress(tmp_path)["stage"] == "resolve"

    write_json(artifacts / "communities.json", {"count": 2, "communities": [{}, {}]})
    assert auto_pipeline.collect_progress(tmp_path)["stage"] == "communities"

    write_json(artifacts / "vectors.json", {"backend": "hash-ngram", "counts": {"nodes": 2}})
    progress = auto_pipeline.collect_progress(tmp_path)
    assert progress["stage"] == "embed"
    assert progress["communities"] == 2
    assert progress["vector_backend"] == "hash-ngram"
    assert progress["vectors"] == {"nodes": 2}


def test_checkpoint_thread_uploads_immediately_and_repeats(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str | None, Path, str, str | None]] = []
    second_upload = threading.Event()

    def fake_upload(repo_name, *, data_dir, token, commit_message=None):
        calls.append((repo_name, data_dir, token, commit_message))
        if len(calls) >= 2:
            second_upload.set()
        return 0

    monkeypatch.setattr(auto_pipeline, "upload_dataset", fake_upload)
    handle = auto_pipeline.start_checkpoint_thread(
        "user/dataset",
        1,
        tmp_path,
        started_at=0,
        token="token",
    )
    assert handle is not None
    assert second_upload.wait(timeout=5)
    auto_pipeline.stop_checkpoint_thread(handle)

    assert len(calls) >= 2
    assert calls[0][0] == "user/dataset"
    assert calls[0][1] == tmp_path
    assert calls[0][2] == "token"
    assert (calls[0][3] or "").startswith("checkpoint ")

    progress = json.loads((tmp_path / "artifacts" / "progress.json").read_text(encoding="utf-8"))
    assert progress["stage"] == "index"
    assert progress["chunks_total"] == 0


def test_checkpoint_thread_disabled_without_interval_or_token(tmp_path: Path) -> None:
    assert auto_pipeline.start_checkpoint_thread("user/dataset", 0, tmp_path, 0, "token") is None
    assert auto_pipeline.start_checkpoint_thread("user/dataset", 3600, tmp_path, 0, "") is None


def test_upload_ignore_excludes_local_model_caches() -> None:
    assert "models/*" in auto_pipeline.UPLOAD_IGNORE
    assert "embeddings/*" in auto_pipeline.UPLOAD_IGNORE
    assert "*.log" in auto_pipeline.UPLOAD_IGNORE
