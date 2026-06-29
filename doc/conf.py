from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
sys.path.insert(0, str(SOURCE))

project = "orc_bound"
author = "XJTU-XGU"

version_file = SOURCE / "orc_bound" / "_version.py"
version_match = re.search(
    r'__version__\s*=\s*"([^"]+)"',
    version_file.read_text(encoding="utf-8"),
)
release = version_match.group(1) if version_match else "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_mock_imports = ["orc_bound._residual_shell_cpp"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_title = f"orc_bound {release} documentation"
