project = 'Graph-informed perturbation prediction'
author = 'Rewant Anand'
release = '0.2'
extensions = ['myst_parser']
source_suffix = {'.md': 'markdown', '.rst': 'restructuredtext'}
master_doc = 'index'
exclude_patterns = ['_build', 'README.md', 'SOURCE_HASHES.json']
html_theme = 'sphinx_rtd_theme'
html_title = project
html_context = {'display_github': True, 'github_user': 'Rewant05',
                'github_repo': 'Graph-Informed-Perturbation-Prediction',
                'github_version': 'main', 'conf_py_path': '/docs/'}
myst_heading_anchors = 3
