"""Topic namespace utilities for AciesOS.

Namespace convention: <namespace>/<name>/<user-defined>
Control subspace:     <namespace>/<name>/ctl/<service>

The namespace is hierarchical (may contain '/') and is set by the deployment.
The name is a flat identifier (no '/') for the service instance.

Usage::

    ns = Namespace('edge-01', 'mic')
    ns = Namespace('edge-01/sensor', 'mic')      # hierarchical namespace
    ns.topic('building', 'a', 'temperature')      # "building/a/temperature"
    ns.topic('audio', prefix=True)                # "<namespace>/mic/audio"
    ns.topic('audio', prefix='org/site-a')        # "org/site-a/audio"
    ns.ctl.kv                                     # "<namespace>/mic/ctl/kv"
    ns.ctl.heartbeat                              # "<namespace>/mic/ctl/heartbeat"
    ns.ctl.route                                  # "<namespace>/mic/ctl/route"
    ns.ctl.io                                     # "<namespace>/mic/ctl/io"
    ns.ctl.schema                                 # "<namespace>/mic/ctl/schema"
    ns.ctl('custom')                              # "<namespace>/mic/ctl/custom"

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

# Additional characters forbidden in concrete identifiers (namespace, name).
# Wildcards are not meaningful in identifiers and would cause silent bugs.
_FORBIDDEN_ID: frozenset[str] = _FORBIDDEN_SEG | frozenset('*$')


def _validate_name(value: str, label: str) -> None:
    """Validate a flat identifier (name) -- no '/' allowed."""
    if not value:
        raise ValueError(f'{label} must not be empty')
    if '/' in value:
        raise ValueError(f"{label} must not contain '/': {value!r}")
    invalid = _FORBIDDEN_ID & set(value)
    if invalid:
        raise ValueError(f'{label} contains forbidden characters {sorted(invalid)}: {value!r}')


def _validate_namespace(value: str) -> None:
    """Validate a hierarchical namespace -- '/' is allowed as separator.

    Each segment between '/' separators is validated individually: must be
    non-empty and free of wildcard/selector characters.
    """
    if not value:
        raise ValueError('namespace must not be empty')
    for seg in value.split('/'):
        if not seg:
            raise ValueError(f'namespace contains empty segment: {value!r}')
        invalid = _FORBIDDEN_ID & set(seg)
        if invalid:
            raise ValueError(f'namespace contains forbidden characters {sorted(invalid)}: {value!r}')


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

        ctl.kv                 # "<namespace>/<name>/ctl/kv"
        ctl.heartbeat          # "<namespace>/<name>/ctl/heartbeat"
        ctl.route              # "<namespace>/<name>/ctl/route"
        ctl.io                 # "<namespace>/<name>/ctl/io"
        ctl.schema             # "<namespace>/<name>/ctl/schema"
        ctl.notify             # "<namespace>/<name>/ctl/notify"
        ctl('my', 'service')   # "<namespace>/<name>/ctl/my/service"
    """

    kv: str
    heartbeat: str
    route: str
    io: str
    schema: str
    notify: str
    base: str

    def __call__(self, *parts: str) -> str:
        for part in parts:
            _validate_part(part)
        return '/'.join([self.base, *parts])


@dataclass
class Namespace:
    """Topic namespace for an AciesApp instance.

    Constructed once at startup from namespace and name; reused across all
    handlers.  The namespace is hierarchical (may contain '/'); the name is
    a flat identifier (no '/').

    Data topics::

        ns.topic('building', 'a', 'temperature')   # "building/a/temperature"
        ns.topic('audio', prefix=True)             # "<ns>/mic/audio"
        ns.topic('audio', prefix='org/site-a')     # "org/site-a/audio"

    Control topics::

        ns.ctl.kv                     # "<ns>/mic/ctl/kv"
        ns.ctl.heartbeat              # "<ns>/mic/ctl/heartbeat"
        ns.ctl.route                  # "<ns>/mic/ctl/route"
        ns.ctl.io                     # "<ns>/mic/ctl/io"
        ns.ctl.schema                 # "<ns>/mic/ctl/schema"
        ns.ctl.notify                 # "<ns>/mic/ctl/notify"
        ns.ctl('custom')              # "<ns>/mic/ctl/custom"
    """

    namespace: str
    name: str
    ctl: CtlTopics = field(init=False)

    def __post_init__(self) -> None:
        _validate_namespace(self.namespace)
        _validate_name(self.name, 'name')
        base = f'{self.namespace}/{self.name}/ctl'
        self.ctl = CtlTopics(
            kv=f'{base}/kv',
            heartbeat=f'{base}/heartbeat',
            route=f'{base}/route',
            io=f'{base}/io',
            schema=f'{base}/schema',
            notify=f'{base}/notify',
            base=base,
        )

    @property
    def base(self) -> str:
        """Return ``<namespace>/<name>`` -- the app's base path in the topic tree."""
        return f'{self.namespace}/{self.name}'

    def topic(self, *parts: str, prefix: bool | str = True) -> str:
        """Construct a topic, optionally prefixed.

        ``prefix=True`` (default) -- prepend ``<namespace>/<name>``.

        ``prefix=''`` or ``prefix=False`` -- no prefix; parts are joined
        as-is for domain-centric topic organisation.

        ``prefix='org/building-a'`` -- prepend a custom path string; the
        string is used verbatim without validation.

        Wildcards ``*``, ``**``, and ``$*`` are allowed in parts.
        """
        for part in parts:
            _validate_part(part)
        if prefix is True:
            return '/'.join([self.base, *parts])
        if prefix:
            return '/'.join([prefix, *parts])
        # no prefix if prefix=False or prefix=''
        return '/'.join(parts)


# ----------------------------- topic arg types --------------------------------


@dataclass(frozen=True)
class Topic:
    """Lazy topic resolved at run() time, optionally prefixed with namespace/name.

    ``prefix=True`` (default) prepends ``<namespace>/<name>`` from the app's
    namespace.  Pass ``prefix=''`` or ``False`` for domain-centric topics,
    or a custom string prefix.  The path may contain ``/`` for convenience.

    Example::

        Topic('audio/raw')                      # "<ns>/mic/audio/raw"
        Topic('**')                             # "<ns>/mic/**"
        Topic('room/5/temperature', prefix='')  # "room/5/temperature"
        Topic('audio', prefix='org/site-a')     # "org/site-a/audio"
    """

    path: str
    prefix: bool | str = True


@dataclass(frozen=True)
class CtlTopic:
    """Lazy control-plane topic resolved to ``<namespace>/<name>/ctl/<path>`` at run() time.

    The path may contain ``/`` for nested control topics.

    Example::

        CtlTopic('kv')          # "<ns>/mic/ctl/kv"
        CtlTopic('heartbeat')   # "<ns>/mic/ctl/heartbeat"
        CtlTopic('route')       # "<ns>/mic/ctl/route"
        CtlTopic('io')          # "<ns>/mic/ctl/io"
        CtlTopic('schema')      # "<ns>/mic/ctl/schema"
        CtlTopic('my/service')  # "<ns>/mic/ctl/my/service"
    """

    path: str


@dataclass(frozen=True)
class OnChange:
    """Lazy notification topic resolved to ``<namespace>/<name>/ctl/notify/<key>`` at run() time.

    Used with ``@app.subscribe`` to react to config changes via kv set/del.

    Example::

        OnChange('start_at')    # "edge-01/mic/ctl/notify/start_at"
        OnChange('speed')       # "edge-01/mic/ctl/notify/speed"
        OnChange('*')           # "edge-01/mic/ctl/notify/*"  (all keys)
    """

    key: str


TopicArg: TypeAlias = str | Topic | CtlTopic | OnChange
