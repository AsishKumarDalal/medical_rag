# G0 — test corpus

Recorded proof that the corpus is reproducible and complete. The corpus itself is
git-ignored; this file is the reference snapshot.

- **Date:** 2026-10-25
- **Command:** `.venv/bin/python test/data/test_data_fetch.py --pause 2`
- **Source:** Wikipedia extracts API (`action=query&prop=extracts&explaintext=1`)
- **Result:** **24 / 24 articles**, 260,462 characters, 8,295–11,995 chars each
- **Manifest:** `test/data/corpus/manifest.json` (per-file sha256, URL, title,
  fetched-at)

## Articles

| id | chars |
|---|---|
| alzheimer_s_disease | 10,895 |
| asthma | 11,765 |
| breast_cancer | 8,295 |
| chronic_kidney_disease | 11,551 |
| chronic_obstructive_pulmonary_disease | 10,800 |
| cirrhosis | 10,838 |
| colorectal_cancer | 8,800 |
| diabetes | 11,768 |
| epilepsy | 11,867 |
| generalized_anxiety_disorder | 11,165 |
| hiv_aids | 9,320 |
| hypertension | 10,834 |
| influenza | 9,541 |
| lung_cancer | 11,798 |
| major_depressive_disorder | 9,090 |
| malaria | 11,181 |
| migraine | 11,669 |
| myocardial_infarction | 11,127 |
| osteoporosis | 11,791 |
| parkinson_s_disease | 11,381 |
| pneumonia | 11,995 |
| sepsis | 10,284 |
| stroke | 11,510 |
| tuberculosis | 11,197 |

## Reproduce

```bash
.venv/bin/python test/data/test_data_fetch.py          # fetch missing articles
.venv/bin/python test/data/test_data_fetch.py --force  # refetch everything
.venv/bin/python -m pytest test/test_fetch.py -q       # offline parse/clean tests
```

The fetcher skips articles already on disk and merges them into the manifest, so
an interrupted run resumes. Wikipedia rate-limits full-article extracts; the
script pauses 2 s between requests and backs off on HTTP 429.
