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
    # "numpydoc",
    # "myst_parser",  # Allow markdown, comment out if myst_nb is enabled
    "myst_nb",  # Execute code blocks in docs
    "sphinx_immaterial",
]

templates_path = ["_templates"]
exclude_patterns = []

myst_enable_extensions = [
    "colon_fence",
]

# Options for autodoc
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
templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "sphinx_immaterial"
html_title = "torch-named-linops"
html_theme_options = {
    "sidebar_hide_name": False,
}
html_css_files = ["codecells.css"]
# html_css_files = []
nb_render_plugin = "default"
