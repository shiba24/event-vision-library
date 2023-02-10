"""Sphinx configuration."""
project = "Event Vision Library"
author = "Shintaro Shiba"
copyright = "2023, Shintaro Shiba"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
