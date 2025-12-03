# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torchlinops

project = "torch-named-linops"
copyright = "2025, Mark Nishimura"
author = "Mark Nishimura"
version = torchlinops.__version__
release = torchlinops.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",  # Enable MathJax for math rendering
    # "numpydoc",
    # "myst_parser",  # Allow markdown, comment out if myst_nb is enabled
    "myst_nb",  # Execute code blocks in docs
    "sphinx_immaterial",
]

templates_path = ["_templates"]
exclude_patterns = []

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",  # Enable $...$ and $$...$$ syntax
    "amsmath",  # Enable LaTeX math environments like \begin{align}...\end{align}
]

# Options for autodoc/autosummary
# Prevent inheriting from torch base modules
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": False,
    "undoc-members": False,
    "private-members": False,
    "inherited-members": False,
    "show-inheritance": False,
}
autosummary_generate = True
add_module_names = False
templates_path = ["_templates"]

# Options for viewcode
viewcode_line_numbers = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ["_static"]
html_theme = "sphinx_immaterial"
html_title = "Torch Named Linops"
html_logo = "_static/logo.svg"
html_theme_options = {
    "repo_url": "https://github.com/nishi951/torch-named-linops",
    "repo_name": " ",  # Only show logo
    "icon": {"repo": "fontawesome/brands/github"},
    "globaltoc_depth": 2,
    "globaltoc_collapse": False,
}
# html_theme_options = {
#     "sidebar_hide_name": False,
# }
html_css_files = ["css/codecells.css"]
# html_css_files = []
nb_render_plugin = "default"
