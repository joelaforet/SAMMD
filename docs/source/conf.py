"""Sphinx configuration for SAMMD documentation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "SAMMD"
author = "Joseph R. Laforet Jr."
copyright = "Joseph R. Laforet Jr."
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\. : | {5,8}: "
copybutton_prompt_is_regexp = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = []
templates_path = ["_templates"]
html_theme = "furo"
html_title = "SAMMD documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/sammd-logo-compact.png"
html_favicon = "_static/sammd-logo-compact.png"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "sammd-logo-compact.png",
    "dark_logo": "sammd-logo-compact.png",
    "light_css_variables": {
        "color-brand-primary": "#C65D20",
        "color-brand-content": "#A94F1B",
        "color-api-name": "#C65D20",
        "color-api-pre-name": "#263238",
    },
    "dark_css_variables": {
        "color-brand-primary": "#E8A256",
        "color-brand-content": "#F5C87A",
    },
}
