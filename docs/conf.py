"""Sphinx configuration."""
import os
import sys
sys.path.insert(0, os.path.abspath('../../src')) 

project = "Event Vision Library"
author = "Team evlib."
copyright = "2023, Team evlib"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]

# Extensions
autosummary_generate = True
autodoc_typehints = "description"
autoclass_content = "class",
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    'member-order': 'bysource',
    'special-members': '__next__,__call__',
    'undoc-members': True,
    # 'exclude-members': '__weakref__'
    # "exclude-members": "with_traceback",
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
}

html_theme = "furo"
templates_path = ['_templates']
exclude_patterns = ['_build', '_templates']
