# Sphinx configuration for orc_bound documentation

project = "orc-bound"
copyright = "2024, ORC-Bound Team"
author = "ORC-Bound Team"
version = "0.1"
release = "0.1.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "basic"
html_static_path = []
