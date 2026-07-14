import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'PCP Pipeline'
copyright = '2026, Author'
author = 'Aurora, Roya, Asif, Ulas'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc', #automatically generated docs from classes/functions docstrings 
    'sphinx.ext.napoleon', #supports numpy style docstrings
    'sphinx.ext.viewcode', #adds links to source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output configuration
html_theme = 'alabaster'
# If no static files exist yet, html_static_path can be empty or omitted.
html_static_path = []
