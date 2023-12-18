# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os

sys.path.insert(0, os.path.abspath('../src'))
import audioman

project = 'audioman'
copyright = f'2023, {audioman.__author__}'
author = audioman.__author__

version = audioman.__version__
release = audioman.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import commonmark

import re
import commonmark

py_attr_re = re.compile(r"\:py\:\w+\:(``[^:`]+``)")

def docstring(app, what, name, obj, options, lines):
    md  = '\n'.join(lines)
    ast = commonmark.Parser().parse(md)
    rst = commonmark.ReStructuredTextRenderer().render(ast)
    lines.clear()
    lines += rst.splitlines()

    for i, line in enumerate(lines):
        while True:
            match = py_attr_re.search(line)
            if match is None:
                break 

            start, end = match.span(1)
            line_start = line[:start]
            line_end = line[end:]
            line_modify = line[start:end]
            line = line_start + line_modify[1:-1] + line_end
        lines[i] = line

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect('autodoc-process-docstring', docstring)
    app.connect("autodoc-skip-member", skip)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx.ext.githubpages',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
