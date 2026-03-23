from __future__ import annotations

import functools
import json
import os
import sys
from typing import Any, Callable

import click

_DEFAULT_CONNECT = f'unix-stream://{os.path.expanduser("~/.acies/zenoh.sock")}'
_DEFAULT_CONFIG = os.path.expanduser('~/.acies/config.json')


def _find_config_path() -> str:
    """Pre-scan sys.argv for --acies-config before Click processes options."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--acies-config' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith('--acies-config='):
            return arg.split('=', 1)[1]
    return _DEFAULT_CONFIG


def _load_defaults(config_path: str) -> dict[str, Any]:
    """Load CLI defaults from a JSON config file.

    Returns empty dict if file does not exist. Keys must match Click parameter
    names (underscores, e.g. 'acies_host').
    """
    path = os.path.expanduser(config_path)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def create_acies_cli(
    configure: Callable[[dict[str, Any]], None], **kwargs: Any
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Return a decorator that turns a function into a Click command.

    Middleware options are injected and consumed before the user's function
    runs; they never appear in the user's kwargs. The configure callable
    receives a dict that is deep-merged into app_state.config.

    Config file precedence: CLI args > config file > built-in defaults.
    Config file keys use Click parameter names (e.g. 'acies_host').
    """
    defaults = _load_defaults(_find_config_path())

    def decorator(user_fn: Callable[..., None]) -> Callable[..., None]:
        @click.command(context_settings={'default_map': defaults}, **kwargs)
        @click.option(
            '--acies-host',
            required=True,
            envvar='ACIES_HOST',
            help='Logical device name for this node (e.g. edge-01).',
        )
        @click.option(
            '--acies-name',
            required=True,
            envvar='ACIES_NAME',
            help='Service name for this app (e.g. mic).',
        )
        @click.option(
            '--acies-state',
            type=click.Choice(['active', 'standby', 'stopped', 'reboot'], case_sensitive=False),
            default='active',
            show_default=True,
            envvar='ACIES_STATE',
            help='Initial service lifecycle state.',
        )
        @click.option(
            '--acies-net-mode',
            type=click.Choice(['client', 'peer'], case_sensitive=False),
            default='client',
            show_default=True,
            envvar='ACIES_NET_MODE',
            help='Zenoh session mode.',
        )
        @click.option(
            '--acies-connect',
            multiple=True,
            default=(_DEFAULT_CONNECT,),
            show_default=True,
            envvar='ACIES_CONNECT',
            help='Endpoints to connect to. May be repeated.',
        )
        @click.option(
            '--acies-listen',
            multiple=True,
            envvar='ACIES_LISTEN',
            help='Endpoints to listen on (peer mode). May be repeated.',
        )
        @click.option(
            '--acies-config',
            default=_DEFAULT_CONFIG,
            show_default=True,
            envvar='ACIES_CONFIG',
            help='Path to JSON config file. CLI args take precedence.',
        )
        @functools.wraps(user_fn)
        def wrapper(
            acies_host: str,
            acies_name: str,
            acies_state: str,
            acies_net_mode: str,
            acies_connect: tuple[str, ...],
            acies_listen: tuple[str, ...],
            acies_config: str,
            **user_kwargs: Any,
        ) -> Any:
            configure(
                {
                    'sys': {
                        'host': acies_host,
                        'name': acies_name,
                        'state': acies_state,
                        'net_mode': acies_net_mode,
                        'connect': list(acies_connect),
                        'listen': list(acies_listen),
                        'config_file': acies_config,
                    }
                }
            )
            return user_fn(**user_kwargs)

        return wrapper

    return decorator
