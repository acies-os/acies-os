[private]
default:
    @just --list

# Format code with ruff
fmt path=".":
    uvx ruff check --select I --fix {{ path }}
    uvx ruff format {{ path }}

# Check code with ruff
check path=".":
    uvx ruff check {{ path }}

# Fix code with ruff
fix path=".":
    uvx ruff check --fix {{ path }}

# delete all *.log files
clean:
    rm -f *.log

DOCS_SOURCE := "source"
DOCS_BUILD := "_build"

# Build HTML docs with Sphinx
build-doc:
    cd docs && uv run sphinx-build -b html {{ DOCS_SOURCE }} {{ DOCS_BUILD }}
    @echo "Docs built at docs/{{ DOCS_BUILD }}/index.html"

live-doc:
    echo $PWD
    uv run docs/watch-doc.py

# Open docs in a browser (Windows/mac/wsl compatible)
view-doc:
    (sleep 1; wslview http://localhost:8000 \
        || xdg-open http://localhost:8000 \
        || powershell.exe -NoProfile start http://localhost:8000 \
        || open http://localhost:8000 \
        || true) &
    python3 -m http.server -d docs/{{ DOCS_BUILD }} 8000
