# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TSB-kit'
copyright = '2024, Paul Boniol, John Paparrizos, Themis Palpanas'
author = 'Paul Boniol, John Paparrizos, Themis Palpanas'
release = '0.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_math_dollar',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_rtd_dark_mode',
    'sphinx.ext.mathjax',
    'sphinx_copybutton'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = ['joblib','six','matplotlib','pandas','arch','tsfresh'
    'hurst','tslearn','cython','scikit-learn','tqdm','scipy',
    'sklearn','stumpy','tensorflow','networkx']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = '../images/logo-tsb-kit.png'

# -- Napolean settings -------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

numpydoc_show_class_members = False

# -- myst_parser settings ----------------------------------------------------
myst_heading_anchors = 3

# -- intersphinx settings ----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'dask': ('https://docs.dask.org/en/stable/', None),
    'numpy': ('https://numpy.org/doc/1.21/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/version/1.3/', None),
    'sklearn': ('https://scikit-learn.org/0.24/', None),
    'statsmodels': ('https://www.statsmodels.org/v0.12.2/', None),
    'pythresh': ('https://pythresh.readthedocs.io/en/latest', None),
    'optuna': ('https://optuna.readthedocs.io/en/v3.1.0/', None),
}

# -- sphinx_rtd_dark_mode settings -------------------------------------------
default_dark_mode = False