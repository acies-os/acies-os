"""Minimal AciesOS node -- template for new applications.

Starts the framework with default system services (heartbeat, ctl) and
no user-defined tasks. Copy this file and add your own handlers.

Usage::

    uv run python examples/minimal.py [--acies-host HOST] [--acies-name NAME]
"""

from __future__ import annotations

from acies.corev2 import AciesApp, setup_logging

app = AciesApp()


@app.cli()
def main() -> None:
    setup_logging(app.name)
    app.run()


if __name__ == '__main__':
    main()
