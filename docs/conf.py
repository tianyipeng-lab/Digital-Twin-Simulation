# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Digital-Twin-Simulation'
# copyright = '2025, Tianyi Peng, George (Zhida) Gui'
author = 'Tianyi Peng, George (Zhida) Gui'
release = '0.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',      
    'sphinx_rtd_theme', 
    "sphinx_design", 
]
myst_enable_extensions = ["colon_fence"]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    # allow sidebar groups to be collapsed/expanded
    "collapse_navigation": True,
    # show two levels: wave → block
    "navigation_depth": 2,
    # show the toctree entries, not just page titles
    "titles_only": False,
}
