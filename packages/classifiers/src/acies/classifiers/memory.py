"""Memory footprint for classifier nodes.

Two views, both in bytes:

- ``weights``: bytes held by the model's parameters and buffers. Excludes the
  torch runtime, activations and allocator overhead, so it is exact and
  repeatable (e.g. to compare one encoder plus N task heads with N end-to-end
  models).
- process memory from psutil: ``rss``, ``uss`` and, on Linux, ``pss``, plus
  ``peak_rss`` (high-water mark, which also covers the transient during model
  load). Sum ``pss`` (``uss`` on macOS) across processes to compare deployments;
  ``rss`` counts shared library pages once per process.

``sample_memory`` takes one sample, logs it, and keeps the latest one in
``sys/memory`` so it can be read on demand (``kv get sys/memory``).

Usage::

    base = sample_memory(ctx, 'baseline')
    model = load_model()
    sample_memory(ctx, 'loaded', model, since=base)
"""

from __future__ import annotations

import logging
import resource
import sys
from typing import Any

import psutil
import torch
from acies.core import AciesContext

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024
_PROC = psutil.Process()


def module_nbytes(module: torch.nn.Module) -> int:
    """Bytes held by the parameters and buffers of ``module`` (shared tensors counted once)."""
    tensors = [*module.parameters(), *module.buffers()]
    return sum(t.numel() * t.element_size() for t in tensors)


def format_bytes(n: int) -> str:
    """Human-readable size, e.g. ``4.0 KiB`` or ``85.3 MiB``."""
    size = float(n)
    for unit in ('B', 'KiB', 'MiB'):
        if size < 1024:
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024
    return f'{size:.1f} GiB'


def process_memory() -> dict[str, int]:
    """Bytes for this process: ``rss``, ``uss``, ``pss`` (Linux only), ``peak_rss``."""
    try:
        info = _PROC.memory_full_info()
    except psutil.Error:  # e.g. no access to smaps; fall back to RSS only
        info = _PROC.memory_info()
    out: dict[str, int] = {'rss': info.rss}
    for name in ('uss', 'pss'):
        val = getattr(info, name, None)
        if val is not None:
            out[name] = val
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB elsewhere
    out['peak_rss'] = peak if sys.platform == 'darwin' else peak * 1024
    return out


def sample_memory(
    ctx: AciesContext,
    tag: str,
    model: torch.nn.Module | None = None,
    since: dict[str, int] | None = None,
) -> dict[str, int]:
    """Sample process (and optionally model weight) memory, log it, and store it in ``sys/memory``.

    ``since`` is an earlier sample; the log line then shows the change in rss/uss/pss.
    """
    snap = process_memory()
    if model is not None:
        snap['weights'] = module_nbytes(model)

    # replace the whole dict so a concurrent `kv get` never sees a partial update
    record: dict[str, Any] = {**snap, 'tag': tag, 'time_ns': ctx.now()}
    ctx.app.config.setdefault('sys', {})['memory'] = record

    parts = [f'{k}={v / _MIB:.1f}MiB' for k, v in snap.items() if k != 'weights']
    if 'weights' in snap:
        parts.append(f'weights={format_bytes(snap["weights"])}')
    if since is not None:
        parts += [
            f'd_{k}={(snap[k] - since[k]) / _MIB:+.1f}MiB' for k in ('rss', 'uss', 'pss') if k in snap and k in since
        ]
    logger.info('mem[%s] %s', tag, ' '.join(parts))
    return snap
