# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import sphinx_rtd_theme
from pathlib import Path

main_path = Path(__file__).parents[2]
sys.path.insert(0, os.path.join(main_path, "src"))
sys.path.insert(0, os.path.join(main_path, "src/odym/modules"))

#sys.path.insert(0, os.path.abspath('..'))
#sys.path.insert(0, os.path.abspath('../..'))


print(sys.path)

project = 'MISO2'
copyright = '2024, Jan Streeck, Benedikt Grammer, Hanspeter Wieland, Barbara Plank, Andre Baumgart, Dominik Wiedenhofer'
author = 'Jan Streeck, Benedikt Grammer, Hanspeter Wieland, Barbara Plank, Andre Baumgart, Dominik Wiedenhofer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'nbsphinx', 'sphinx_rtd_theme'
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
nbsphinx_allow_errors = True

nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path('stanford-theme')]
html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']

try:
    import config.MISO2_config
    print("Succesfully imported MISO2_config")
except ImportError as e:
    print(f"Error importing module: {e}")