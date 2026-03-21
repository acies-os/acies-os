"""SubscriberSpec, ScheduleSpec, ServiceSpec, Job — core data structures.

Three distinct frozen dataclasses, one per trigger kind. Their union is the
TaskSpec type alias. The sum type pattern makes invalid states unrepresentable:
a ScheduleSpec cannot have topics; a SubscriberSpec cannot have an interval.
No TaskKind enum is needed — the type itself is the discriminant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class SubscriberSpec:
    name: str
    fn: Callable
    topics: tuple[str, ...]
    msg_type: type | None = None  # extracted from fn annotation at decoration time


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    fn: Callable
    interval: float


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    fn: Callable
    topics: tuple[str, ...]
    msg_type: type | None = None  # extracted from fn annotation at decoration time


TaskSpec = SubscriberSpec | ScheduleSpec | ServiceSpec


@dataclass
class Job:
    """One unit of work, created at runtime from a TaskSpec.

    All three task kinds flow through the executor. The router passes raw bytes
    straight through — decoding happens in the dispatch function (worker thread).
    ServiceSpec jobs carry send_bytes so dispatch can encode and deliver the
    reply after the handler returns, without the handler knowing about the
    underlying query mechanism.
    """

    spec: TaskSpec
    raw: bytes | None  # raw msgpack bytes from transport; None for ScheduleSpec jobs
    deadline: float = 0.0  # seconds (monotonic); 0.0 = no deadline, FIFO ordering
    created_at: float = field(default_factory=time.monotonic)
    reply_fn: Callable[[bytes], None] | None = None  # only set for ServiceSpec jobs
