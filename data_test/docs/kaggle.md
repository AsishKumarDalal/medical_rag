# Running data_test on Kaggle

## Requirements

- Kaggle notebook with **Internet ON**.
- GPU runtime (T4 ×2 or P100) recommended.
- `HF_TOKEN` stored as a Kaggle Secret (write access for dataset upload).

## One-command run

```bash
cd /kaggle/working/medical_rag && git pull && rm -rf .venv data_test/.prepared && PYTHONPATH= python data_test/auto_pipeline.py --min-articles 50
```

This performs:

1. venv creation (`uv venv --seed`), pip bootstrap, dependency install
2. Ollama install (zstd or standalone binary fallback) and model pulls
3. corpus fetch/resume to at least 50 articles
4. extract → graph → resolve → communities → embed
5. bundle verification
6. upload/update of `<username>/medical-graphrag-data-test`
7. demo question

## If the pipeline is interrupted

Extraction is checkpointed. Resume without refetching or rebuilding:

```bash
cd /kaggle/working/medical_rag
PYTHONPATH= .venv/bin/python data_test/auto_pipeline.py --retry-failed --skip-models --skip-corpus
```

## Apply Ollama GPU settings to an existing server

```bash
cd /kaggle/working/medical_rag
OLLAMA_RESTART=1 USE_OLLAMA=1 .venv/bin/python data_test/setup_ollama.py
```

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `No module named pip` in `.venv` | `uv venv` without `--seed` | rerun `prepare_environment.py`; it bootstraps pip |
| `Error in sitecustomize ... wrapt` | Kaggle `PYTHONPATH` | run with `PYTHONPATH=`; scripts sanitize it |
| Ollama installer: `requires zstd` | missing `zstd` | `setup_ollama.py` installs it or uses the binary fallback |
| `ollama` not found after install | installed to `~/.local/bin` | `setup_ollama.py` extends `PATH` |
| model only on one GPU | 4B Q4 fits on one T4 | expected; `OLLAMA_SCHED_SPREAD=1` is already set |
| upload skipped | no `HF_TOKEN` | add the Kaggle Secret or export `HF_TOKEN` |

## Useful options

```bash
--min-articles 50     corpus target
--retry-failed        re-ask failed chunks
--workers 1           keep one extraction worker
--skip-models         do not download GGUF/BGE weights
--skip-corpus         reuse the existing corpus
--skip-build          skip indexing
--skip-push           do not upload
--skip-demo           do not run the demo
--repo-id NAME        override the dataset repo name
```

## Token handling

`prepare_environment.py` reads `HF_TOKEN` from the environment or Kaggle secrets
and adds it to `.env` without overwriting existing settings. Tokens are never
printed or uploaded.
