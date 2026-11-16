"""Adopt from: https://github.com/pola-rs/polars/blob/main/py-polars/docs/source/conf.py"""

from livereload import Server, shell

# from source.conf import html_static_path, templates_path

if __name__ == '__main__':
    # establish a local docs server
    svr = Server()

    # command to rebuild the docs
    # refresh_docs = shell('just build-doc')
    refresh_docs = shell('uv run sphinx-build -b html docs/source docs/_build')

    # watch for source file changes and trigger rebuild/refresh
    svr.watch('**/*.py', refresh_docs, delay=1)
    svr.watch('**/*.rst', refresh_docs, delay=1)
    svr.watch('*.md', refresh_docs, delay=1)
    # svr.watch('source/reference/*', refresh_docs, delay=1)
    # for path in html_static_path + templates_path:
    #     svr.watch(f'source/{path}/*', refresh_docs, delay=1)

    # path from which to serve the docs
    svr.serve(root='docs/_build')
