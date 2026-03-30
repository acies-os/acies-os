# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'acies-vehicle-classifier'
copyright = '2025, Shangchen Wu, Jinyang Li'
author = 'Shangchen Wu, Jinyang Li'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    # 'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
]

html_theme = 'pydata_sphinx_theme'
# html_theme = "sphinx_rtd_theme"  # alternative

# Better API pages
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    # 'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}
autoclass_content = 'class'  # class docstring + __init__ docstring
always_document_param_types = True
typehints_fully_qualified = False
# Show types in the description block (clearer than inline)
typehints_format = 'short'  # (or "fully-qualified")

# Napoleon (Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx (clickable types to Python/NumPy docs)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# PyData theme options (nice sidebar + GitHub link)
html_show_sourcelink = False
html_theme_options = {
    'navigation_depth': 4,
    'show_prev_next': False,
    'logo': {'text': 'acies-vehicle-classifier'},
    'github_url': 'https://github.com/acies-os/vehicle-classifier',
    'show_toc_level': 3,
}

# Configure TOC to include class methods
toc_object_entries_show_parents = 'hide'
autodoc_class_signature = 'separated'
