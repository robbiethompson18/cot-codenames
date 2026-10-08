# Code smells

- 2026-10-08: `scripts/stego_report/*.py` are quick HTML report builders written during the first
  stego spike. They build SVG by string concatenation, are exempt from several lint rules in
  `pyproject.toml`, and `build_report.py` / `build_essays.py` hard-code run names and essay picks.
  Rewrite or delete once the report format settles.
- 2026-10-08: `stego/remonitor.py` calls `TinkerBackend.choose` on an object built with `__new__` to
  skip `__init__`. Pull the label-probability call out into a function both can use.
