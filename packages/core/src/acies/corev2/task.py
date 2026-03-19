"""TaskKind, TaskSpec, Job — core data structures.

TaskKind describes the trigger mechanism (not topology). Whether a handler
is a sink or transform depends on whether it calls ctx.publish(), not on
the decorator used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TaskKind(Enum):
    # triggered by an incoming message on a topic
    SUBSCRIBE = 'subscribe'
    # triggered by a timer
    SCHEDULE = 'schedule'
    # triggered by an external event (sensor thread)
    PRODUCE = 'produce'
    # triggered by an RPC query (zenoh queryable)
    SERVICE = 'service'


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of a task, registered at decoration time."""

    name: str
    kind: TaskKind
    fn: Callable[..., None]
    # non-empty for SUBSCRIBE and SERVICE
    topics: tuple[str, ...] = ()
    # set for SCHEDULE
    interval: float | None = None


@dataclass
class Job:
    """One unit of work, created at runtime from a TaskSpec.

    All four task kinds flow through the executor. SERVICE jobs carry a
    reply_fn so the executor can send the reply after the handler returns,
    without the handler needing to know about the underlying query mechanism.
    """

    spec: TaskSpec
    msg: Any | None  # AciesMsg | None; None for SCHEDULE and PRODUCE
    created_at: float = field(default_factory=time.monotonic)
    reply_fn: Callable[[Any], None] | None = None  # only set for SERVICE jobs
