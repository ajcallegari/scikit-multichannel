# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# Source code dir relative to this file
# sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'skmultichannel')))


# -- Project information -----------------------------------------------------

project = 'scikit-multichannel'
copyright = '2021, A. John Callegari Jr.'
author = 'A. John Callegari Jr.'

# The full version, including alpha/beta/rc tags
release = 'v0.4-alpha'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.linkcode']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Tell autosummary to make rst stub files
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['../skmultichannel/tests']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# function that allows the linkcode extension to find github pages
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    base = "https://github.com/ajcallegari/scikit-multichannel/blob/master"
    return base + "/%s.py" % filename
