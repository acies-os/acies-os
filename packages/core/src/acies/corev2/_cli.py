from __future__ import annotations

import functools
from typing import Any, Callable

import click


def make_cli_decorator(
    configure: Callable[[dict[str, Any]], None], **kwargs: Any
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Return a decorator that turns a function into a Click command.

    Middleware options (--acies-host, --acies-name) are injected and consumed
    before the user's function runs; they never appear in the user's kwargs.
    The configure callable receives the parsed middleware values and applies
    them to the app (e.g. deep-merging into app_state.config['sys']).
    """

    def decorator(user_fn: Callable[..., None]) -> Callable[..., None]:
        @click.command(**kwargs)
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
        @functools.wraps(user_fn)
        def wrapper(acies_host: str, acies_name: str, **user_kwargs: Any) -> Any:
            configure({'sys': {'host': acies_host, 'name': acies_name}})
            return user_fn(**user_kwargs)

        return wrapper

    return decorator
