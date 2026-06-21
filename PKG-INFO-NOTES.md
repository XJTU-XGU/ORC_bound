# Maintainer Notes

This source tree is intentionally independent from the surrounding research
scripts. Existing files outside this package are not required for installation.

Suggested pre-release checks:

```bash
python -m pip install -e .
python examples/basic_usage.py
python -m pytest
python -m build
twine check dist/*
```

If publishing binary wheels, use CI such as `cibuildwheel` to build per-platform
wheels. Otherwise PyPI users will build the extension from source.


