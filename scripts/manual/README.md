# Manual scripts

These are **not** automated tests. They are ad-hoc scripts that hit the network
(yfinance) and, in some cases, download the ~400MB Kronos model at import time.
They live here rather than at the repo root so `pytest` collection does not pick
them up and start downloading models.

Run them by hand from the repo root when you want them:

```bash
python scripts/manual/test_all.py
python scripts/manual/test_infy.py
python scripts/manual/test_multi_symbol.py
```

Real unit tests belong in `tests/` and must not require network access.
