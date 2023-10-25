# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Mock imports for autodoc --- all packages your project uses that do not come pre-installed with python
autodoc_mock_imports = [
    'tqdm',
    'scipy',
    'numpy',
    'astropy',
    'photutils',
    'radio_beam',
    'matplotlib',
    'sslf'
    ]

from unittest.mock import MagicMock
for mod in autodoc_mock_imports:
    sys.modules[mod] = MagicMock()

sys.path.insert(0, os.path.abspath('..'))

import sphinx_rtd_theme

project = 'zfinder'
copyright = '2023, Daniel Lyon'
author = 'Daniel Lyon'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.autodoc", "sphinx.ext.viewcode", "sphinx_rtd_theme"]    

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tests', 'setup.py']

source_suffix = '.rst'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']