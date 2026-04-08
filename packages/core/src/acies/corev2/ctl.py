"""Control plane helpers for AciesOS.

Convenience functions for building and sending KV operations to remote nodes.
Intended for control plane services (gateway, controller, etc.) — not for
normal data-plane handlers.

Usage::

    from acies.corev2.ctl import kv_get, kv_set, kv_del, kv_call

    # build ops (no I/O)
    ops = [
        kv_set('scene', value=scene),
        kv_set('run', value=run_id),
        kv_get('sys', 'state'),
    ]

    # send in one round-trip
    results = kv_call(ctx, 'rs1/mic', ops)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .context import AciesContext
from .msg import (
    AciesDel,
    AciesGet,
    AciesKvRequest,
    AciesKvResponse,
    AciesResult,
    AciesSet,
    KvEntry,
)

logger = logging.getLogger(__name__)


def kv_get(*key: str) -> AciesGet:
    """Build a get operation for the given key path.

    Example::

        kv_get('scene')            # AciesGet(key=['scene'])
        kv_get('sys', 'state')     # AciesGet(key=['sys', 'state'])
    """
    return AciesGet(key=list(key))


def kv_set(*key: str, value: Any) -> AciesSet:
    """Build a set operation for the given key path and value.

    Example::

        kv_set('scene', value='2024-08-06-GQ')
        kv_set('sys', 'state', value='active')
    """
    return AciesSet(key=list(key), value=value)


def kv_del(*key: str) -> AciesDel:
    """Build a delete operation for the given key path.

    Example::

        kv_del('threshold')
        kv_del('model', 'weights')
    """
    return AciesDel(key=list(key))


def kv_call(
    ctx: AciesContext,
    target: str,
    ops: Sequence[KvEntry],
    *,
    timeout: float = 1.0,
) -> list[AciesResult]:
    """Send KV operations to a target node and return results.

    Constructs an ``AciesKvRequest``, sends it to ``<target>/ctl/kv``,
    and returns the list of ``AciesResult`` (``Ok`` or ``Err``) in the
    same order as ``ops``.

    Returns an empty list if the target does not respond within *timeout*.

    Example::

        results = kv_call(ctx, 'rs1/mic', [
            kv_set('scene', value='2024-08-06-GQ'),
            kv_set('run', value=29),
        ])
    """
    req = AciesKvRequest(source=target, timestamp=ctx.now(), ops=ops)
    resp = ctx.query(f'{target}/ctl/kv', req, timeout=timeout, reply_type=AciesKvResponse)
    if resp is None:
        logger.error('no response from %s/ctl/kv (timeout=%.1fs)', target, timeout)
        return []
    if not isinstance(resp, AciesKvResponse):
        logger.error('unexpected response type from %s/ctl/kv: %s', target, type(resp))
        return []
    return list(resp.results)
