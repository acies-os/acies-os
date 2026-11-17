# Sphinx configuration for the Acies-OS documentation website.
# Source directory: docs/   Build: `just build-doc` -> docs/_build/index.html
#
# The pages under integration/ are plain Markdown so they also read well on GitHub.
# Two small hooks below adapt them for the website:
#   * GitHub alerts ("> [!NOTE]") become Sphinx admonitions.
#   * An index.md that links its pages as "NN-name.md" gets a hidden toctree,
#     so those pages appear in the sidebar navigation.

import re

project = 'acies-os'
author = 'Acies-OS team'
copyright = '2026, Acies-OS team'
release = '0.3.0'

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
]
source_suffix = {'.md': 'markdown'}
root_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_enable_extensions = ['colon_fence', 'deflist']
myst_heading_anchors = 3

html_theme = 'pydata_sphinx_theme'
html_title = 'Acies-OS'
html_show_sourcelink = False
html_static_path = ['_static']
html_css_files = ['custom.css']
html_context = {'default_mode': 'auto'}   # follow the reader's system setting
html_theme_options = {
    'navbar_align': 'left',
    'navbar_center': [],          # no top-level page links; the sidebar has the navigation
    'navbar_end': ['search-button-field', 'theme-switcher'],
    'navbar_persistent': [],
    'secondary_sidebar_items': {'**': ['page-toc'], 'index': []},   # home page: no 'On this page'
    'navigation_depth': 3,
    'show_toc_level': 2,
    'show_prev_next': True,
    'pygments_light_style': 'github-light-high-contrast',
    'pygments_dark_style': 'github-dark',
    'footer_start': ['copyright'],
    'footer_center': [],
    'footer_end': [],
}

_ALERT = re.compile(r'^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\n((?:>.*\n?)*)', re.M)
_PAGE_LINK = re.compile(r'\]\((\d\d-[\w-]+)\.md\)')


def _alert_to_admonition(match: re.Match) -> str:
    kind = match.group(1).lower()
    body = re.sub(r'^> ?', '', match.group(2), flags=re.M).rstrip('\n')
    return f'```{{{kind}}}\n{body}\n```\n'


def _adapt_markdown(app, docname, source):
    text = _ALERT.sub(_alert_to_admonition, source[0])
    if docname.endswith('index') and '{toctree}' not in text:
        pages = list(dict.fromkeys(_PAGE_LINK.findall(text)))
        if pages:
            text += '\n\n```{toctree}\n:hidden:\n\n' + '\n'.join(pages) + '\n```\n'
    source[0] = text


def setup(app):
    app.connect('source-read', _adapt_markdown)
