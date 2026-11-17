[private]
default:
    @just -f {{ justfile() }} --list

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

# Sync all workspace packages and dev groups from the lockfile (no upgrades)
deps-dev:
    uv sync --all-packages --all-groups

# Bump git deps (foundation-sense, rawshake) in uv.lock, then sync
deps-upgrade:
    uv sync --all-packages \
        --upgrade-package foundation-sense \
        --upgrade-package rawshake

DOCS_BUILD := "docs/_build"

# Build the documentation website (HTML) into docs/_build
build-doc:
    uv run --frozen --only-group docs sphinx-build -b html docs {{ DOCS_BUILD }}
    @echo "Docs built at {{ DOCS_BUILD }}/index.html"

# Serve the docs with live reload while editing (http://localhost:8000)
live-doc:
    uv run --frozen --only-group docs sphinx-autobuild docs {{ DOCS_BUILD }} --port 8000

# Serve the built docs and open them in a browser
view-doc:
    (sleep 1; wslview http://localhost:8000 \
        || xdg-open http://localhost:8000 \
        || powershell.exe -NoProfile start http://localhost:8000 \
        || open http://localhost:8000 \
        || true) &
    python3 -m http.server -d {{ DOCS_BUILD }} 8000
