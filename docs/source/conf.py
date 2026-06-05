"""Sphinx configuration for SAMMD documentation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "SAMMD"
author = "SAMMD contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = []
html_theme = "alabaster"
html_title = "SAMMD documentation"
