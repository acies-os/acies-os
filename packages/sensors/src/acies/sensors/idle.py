"""Idle node for AciesOS — no workload, just the default system services.

Useful for profiling framework overhead (heartbeat, ctl services, executor)
in isolation.

Usage::

    acies-idle [--acies-namespace NS] [--acies-name NAME]
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
