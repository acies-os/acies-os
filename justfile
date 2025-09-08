[private]
default:
    @just --list

# delete all *.log files
clean:
    rm -f *.log

RUNCMD := `command -v uv || command -v rye || (echo "Please install uv" >&2 && exit 1)`

DOCS_SOURCE := "sphinx"
DOCS_BUILD := "docs"

# Build HTML docs with Sphinx
build-doc:
    {{ RUNCMD }} run sphinx-build -b html {{ DOCS_SOURCE }} {{ DOCS_BUILD }}/html
    @echo "Docs built at {{ DOCS_BUILD }}/html/index.html"

# Open docs in a browser (Windows/mac/wsl compatible)
view-doc:
    (sleep 1; wslview http://localhost:8000 \
        || xdg-open http://localhost:8000 \
        || powershell.exe -NoProfile start http://localhost:8000 \
        || open http://localhost:8000 \
        || true) &
    python3 -m http.server -d {{ DOCS_BUILD }}/html 8000