"""Transport backends for AciesOS.

Routing between transports is prefix-based: the Router uses prefix rules
to decide which transport handles a given topic.

Concrete backends:
  ZenohTransport     -- cross-node pub/sub + queryable; handles bare topics (default)
  LocalTransport     -- in-process queue; used for tests
  WebSocketTransport -- browser/UI; topic prefix 'ws://'
"""

from ._base import MessageHandler, ReplyCallback, Transport
from .local import LocalTransport
from .websocket import WebSocketTransport
from .zenoh import ZenohTransport

__all__ = [
    'MessageHandler',
    'ReplyCallback',
    'Transport',
    'LocalTransport',
    'WebSocketTransport',
    'ZenohTransport',
]
