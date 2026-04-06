"""Logging setup for AciesOS applications.

Configures a root logger with two handlers:
  - Console (StreamHandler): INFO and above
  - File (RotatingFileHandler): DEBUG and above, saved to
    ~/.acies/logs/<namespace>/<name>.log

Per-logger level overrides can be set via the ``ACIES_LOG`` environment variable::

    ACIES_LOG=acies.corev2=INFO,acies.sensors.geo=DEBUG acies-geo ...

Usage::

    from acies.corev2 import setup_logging

    setup_logging('geo')                          # -> ~/.acies/logs/geo.log
    setup_logging('geo', namespace='edge-01')     # -> ~/.acies/logs/edge-01/geo.log
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import pathlib


def setup_logging(name: str, namespace: str | None = None) -> None:
    """Configure root logger with console and rotating file handlers.

    Handlers:
      - Console (stderr): INFO and above.
      - File: DEBUG and above. Rotates at 50 MB, keeps 20 backups (~1 GB total).

    Log file path:
      - ``setup_logging('geo')`` -> ``~/.acies/logs/geo.log``
      - ``setup_logging('geo', namespace='edge-01/sensor')``
        -> ``~/.acies/logs/edge-01/sensor/geo.log``

    Parent directories are created automatically.

    Format::

        <level>yyyymmdd HH:MM:SS.mmm000 <thread> <file>:<line>] <message>

    where <level> is the first character of the level name (D/I/W/E/C) and
    the fractional seconds field is milliseconds zero-padded to 6 digits.

    Per-logger level overrides are applied from the ``ACIES_LOG`` environment
    variable (comma-separated ``name=LEVEL`` pairs) after the root logger is
    configured, so they take precedence over the defaults.

    Args:
        name: Log file base name (e.g. 'geo').
        namespace: Optional hierarchical namespace (e.g. 'edge-01/sensor').
            When provided, logs are written under a matching directory hierarchy.
    """
    log_dir = pathlib.Path.home() / '.acies' / 'logs'
    if namespace:
        log_dir = log_dir / namespace

    fmt = logging.Formatter(
        '%(levelname)-.1s%(asctime)s.%(msecs)06d %(thread)d %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y%m%d %H:%M:%S',
    )

    log_path = log_dir / f'{name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=20
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # suppress middleware debug noise by default; override via ACIES_LOG
    logging.getLogger('acies.corev2').setLevel(logging.INFO)

    for entry in os.environ.get('ACIES_LOG', '').split(','):
        entry = entry.strip()
        if '=' not in entry:
            continue
        logger_name, level = entry.split('=', 1)
        logging.getLogger(logger_name.strip()).setLevel(level.strip())
