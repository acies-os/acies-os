"""Topic namespace utilities for AciesOS.

Namespace convention: <host>/<name>/<user-defined>
Control subspace:     <host>/<name>/ctl/<service>

Usage::

    ns = Namespace('edge-01', 'mic')
    ns.topic('building', 'a', 'temperature')   # "building/a/temperature"
    ns.topic('audio', prefix=True)             # "edge-01/mic/audio"
    ns.topic('audio', prefix='org/site-a')     # "org/site-a/audio"
    ns.ctl.kv                                  # "edge-01/mic/ctl/kv"
    ns.ctl.heartbeat                           # "edge-01/mic/ctl/heartbeat"
    ns.ctl.route                               # "edge-01/mic/ctl/route"
    ns.ctl.io                                  # "edge-01/mic/ctl/io"
    ns.ctl.schema                              # "edge-01/mic/ctl/schema"
    ns.ctl('custom')                           # "edge-01/mic/ctl/custom"

See::

    https://github.com/eclipse-zenoh/roadmap/blob/main/rfcs/ALL/Key%20Expressions.md
    https://zenoh-python.readthedocs.io/en/1.8.0/
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TypeAlias

# Selector characters — never valid in a key expression.
_FORBIDDEN_SEG: frozenset[str] = frozenset('?#')

# Additional characters forbidden in concrete identifiers (host, name).
# Wildcards are not meaningful in identifiers and would cause silent bugs.
_FORBIDDEN_ID: frozenset[str] = _FORBIDDEN_SEG | frozenset('*$')


def _validate_id(value: str, label: str) -> None:
    """Validate a concrete identifier (host or name) — wildcards not allowed."""
    if not value:
        raise ValueError(f'{label} must not be empty')
    if '/' in value:
        raise ValueError(f"{label} must not contain '/': {value!r}")
    invalid = _FORBIDDEN_ID & set(value)
    if invalid:
        raise ValueError(f'{label} contains forbidden characters {sorted(invalid)}: {value!r}')


def _validate_part(part: str) -> None:
    """Validate a zenoh key expression segment.

    Allowed wildcard forms:
      ``*``    — standalone segment, matches one non-empty, non-'/' chunk.
      ``**``   — standalone segment, matches any number of chunks.
      ``$*``   — infix pattern, may be combined with other chars (e.g. ``thermo$*``).

    Selector characters ``?`` and ``#`` are always rejected.
    """
    if not part:
        raise ValueError('topic part must not be empty')
    if '/' in part:
        raise ValueError(f"topic part must not contain '/': {part!r}")
    if part in ('*', '**'):
        return
    invalid = _FORBIDDEN_SEG & set(part)
    if invalid:
        raise ValueError(f'topic part contains forbidden characters {sorted(invalid)}: {part!r}')
    # Walk the segment to validate * and $ usage.
    i = 0
    while i < len(part):
        ch = part[i]
        if ch == '$':
            if i + 1 >= len(part) or part[i + 1] != '*':
                raise ValueError(f"'$' must be followed by '*' in topic part: {part!r}")
            i += 2
        elif ch == '*':
            raise ValueError(f"'*' must be a standalone segment or follow '$' in: {part!r}")
        else:
            i += 1


def matches(pattern: str, topic: str) -> bool:
    """Return True if topic matches pattern.

    Supports zenoh-style wildcards:
      *   — exactly one chunk (non-empty sequence of non-'/' chars)
      **  — any number of chunks, including zero (may span multiple '/' separators)
    Exact match always works.
    """
    if pattern == topic:
        return True
    regex = ''
    i = 0
    while i < len(pattern):
        if pattern[i : i + 2] == '**':
            regex += '.*'
            i += 2
        elif pattern[i] == '*':
            regex += '[^/]+'
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return bool(re.fullmatch(regex, topic))


@dataclass
class CtlTopics:
    """Default control topic strings for a given app.

    Attribute access for known control topics; callable for custom ones::

        ctl.kv                 # "<host>/<name>/ctl/kv"
        ctl.heartbeat          # "<host>/<name>/ctl/heartbeat"
        ctl.route              # "<host>/<name>/ctl/route"
        ctl.io                 # "<host>/<name>/ctl/io"
        ctl.schema             # "<host>/<name>/ctl/schema"
        ctl('my', 'service')   # "<host>/<name>/ctl/my/service"
    """

    kv: str
    heartbeat: str
    route: str
    io: str
    schema: str
    base: str

    def __call__(self, *parts: str) -> str:
        for part in parts:
            _validate_part(part)
        return '/'.join([self.base, *parts])


@dataclass
class Namespace:
    """Topic namespace for an AciesApp instance.

    Constructed once at startup from host and name; reused across all handlers.

    Data topics::

        ns.topic('building', 'a', 'temperature')   # "building/a/temperature"
        ns.topic('audio', prefix=True)             # "edge-01/mic/audio"
        ns.topic('audio', prefix='org/site-a')     # "org/site-a/audio"

    Control topics::

        ns.ctl.kv                     # "edge-01/mic/ctl/kv"
        ns.ctl.heartbeat              # "edge-01/mic/ctl/heartbeat"
        ns.ctl.route                  # "edge-01/mic/ctl/route"
        ns.ctl.io                     # "edge-01/mic/ctl/io"
        ns.ctl.schema                 # "edge-01/mic/ctl/schema"
        ns.ctl('custom')              # "edge-01/mic/ctl/custom"
    """

    host: str
    name: str
    ctl: CtlTopics = field(init=False)

    def __post_init__(self) -> None:
        _validate_id(self.host, 'host')
        _validate_id(self.name, 'name')
        base = f'{self.host}/{self.name}/ctl'
        self.ctl = CtlTopics(
            kv=f'{base}/kv',
            heartbeat=f'{base}/heartbeat',
            route=f'{base}/route',
            io=f'{base}/io',
            schema=f'{base}/schema',
            base=base,
        )

    @property
    def base(self) -> str:
        """Return ``<host>/<name>`` — the app's base path in the topic tree."""
        return f'{self.host}/{self.name}'

    def topic(self, *parts: str, prefix: bool | str = True) -> str:
        """Construct a topic, optionally prefixed.

        ``prefix=True`` (default) — prepend ``<host>/<name>``.

        ``prefix=''`` or ``prefix=False`` — no prefix; parts are joined
        as-is for domain-centric topic organisation.

        ``prefix='org/building-a'`` — prepend a custom path string; the
        string is used verbatim without validation.

        Wildcards ``*``, ``**``, and ``$*`` are allowed in parts.
        """
        for part in parts:
            _validate_part(part)
        if prefix is True:
            return '/'.join([self.host, self.name, *parts])
        if prefix:
            return '/'.join([prefix, *parts])
        # no prefix if prefix=False or prefix=''
        return '/'.join(parts)


# ----------------------------- topic arg types --------------------------------


@dataclass(frozen=True)
class Topic:
    """Lazy topic resolved at run() time, optionally prefixed with host/name.

    ``prefix=True`` (default) prepends ``<host>/<name>`` from the app's
    namespace.  Pass ``prefix=''`` or ``False`` for domain-centric topics,
    or a custom string prefix.  The path may contain ``/`` for convenience.

    Example::

        Topic('audio/raw')                      # "edge-01/mic/audio/raw"
        Topic('**')                             # "edge-01/mic/**"
        Topic('room/5/temperature', prefix='')  # "room/5/temperature"
        Topic('audio', prefix='org/site-a')     # "org/site-a/audio"
    """

    path: str
    prefix: bool | str = True


@dataclass(frozen=True)
class CtlTopic:
    """Lazy control-plane topic resolved to ``<host>/<name>/ctl/<path>`` at run() time.

    The path may contain ``/`` for nested control topics.

    Example::

        CtlTopic('kv')          # "edge-01/mic/ctl/kv"
        CtlTopic('heartbeat')   # "edge-01/mic/ctl/heartbeat"
        CtlTopic('route')       # "edge-01/mic/ctl/route"
        CtlTopic('io')          # "edge-01/mic/ctl/io"
        CtlTopic('schema')      # "edge-01/mic/ctl/schema"
        CtlTopic('my/service')  # "edge-01/mic/ctl/my/service"
    """

    path: str


TopicArg: TypeAlias = str | Topic | CtlTopic
