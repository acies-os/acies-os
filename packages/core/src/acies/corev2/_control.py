"""Default system control handlers — registered automatically by AciesApp.

Currently provides:
  - Heartbeat: periodic pub on ns.ctl.heartbeat so other nodes can detect
    liveness. Interval is configurable; default is 5 seconds.
"""

from __future__ import annotations

from .context import AciesContext
from .msg import AciesHeartbeat
from .task import ScheduleSpec

_DEFAULT_HEARTBEAT_INTERVAL: float = 5.0


def _heartbeat(ctx: AciesContext) -> None:
    state = ctx.app.config.get('sys', {}).get('state', 'active')
    ctx.publish(ctx.ns.ctl.heartbeat, AciesHeartbeat(state=state))


def make_heartbeat_spec(interval: float = _DEFAULT_HEARTBEAT_INTERVAL) -> ScheduleSpec:
    """Return a ScheduleSpec that publishes AciesHeartbeat at *interval* seconds."""
    return ScheduleSpec(name='_heartbeat', fn=_heartbeat, interval=interval)
