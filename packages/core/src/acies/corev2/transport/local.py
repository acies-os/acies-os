"""LocalTransport -- in-process queue-based transport for testing.

Logger: acies.corev2.transport.local

Always handles all topics. Add to Router last so prefixed transports like
WebSocketTransport take priority.

Query flow
----------
1. Caller calls query(topic, raw, timeout).
2. LocalTransport checks that the topic is advertised.
3. Creates a reply_fn closure backed by a threading.Event.
4. Puts (topic, raw, reply_fn) directly into the inbound queue.
5. Receiver thread delivers it via on_message(topic, raw, reply_fn).
6. Router creates a Job with reply_fn; dispatch encodes the result and
   calls job.reply_fn(encoded_bytes).
7. reply_fn sets result[0] and the event; query() unblocks and returns
   the encoded bytes.
"""

from __future__ import annotations

import logging
import queue
import threading

from .._concurrency import SENTINEL, Sentinel
from ..namespace import matches
from ._base import MessageHandler, ReplyCallback

logger = logging.getLogger(__name__)


class LocalTransport:
    def __init__(self) -> None:
        self._subscriptions: set[str] = set()
        self._advertisers: set[str] = set()
        self._queue: queue.SimpleQueue[tuple[str, bytes, ReplyCallback | None] | Sentinel] = queue.SimpleQueue()
        self._on_message: MessageHandler | None = None
        self._thread: threading.Thread | None = None

    def start(self, on_message: MessageHandler) -> None:
        self._on_message = on_message
        self._thread = threading.Thread(target=self._receiver_loop, name='local-transport', daemon=True)
        self._thread.start()
        logger.debug('started')

    def stop(self) -> None:
        logger.debug('stop requested')
        self._queue.put(SENTINEL)
        if self._thread:
            self._thread.join()
        logger.debug('stopped')

    def abort(self) -> None:
        n_discarded = 0
        while True:
            try:
                _ = self._queue.get(block=False)
                n_discarded += 1
            except queue.Empty:
                break
        logger.debug('abort: discarded %d queued messages', n_discarded)
        self._queue.put(SENTINEL)
        if self._thread:
            self._thread.join()
        logger.debug('aborted')

    def subscribe(self, topic: str) -> None:
        self._subscriptions.add(topic)
        logger.debug('subscribed %r', topic)

    def unsubscribe(self, topic: str) -> None:
        self._subscriptions.discard(topic)
        logger.debug('unsubscribed %r', topic)

    def advertise(self, topic: str) -> None:
        self._advertisers.add(topic)
        logger.debug('advertised %r', topic)

    def unadvertise(self, topic: str) -> None:
        self._advertisers.discard(topic)
        logger.debug('unadvertised %r', topic)

    def publish(self, topic: str, raw: bytes) -> None:
        """Deliver raw bytes if topic matches any active subscription."""
        if any(matches(pattern, topic) for pattern in self._subscriptions):
            self._queue.put((topic, raw, None))
            logger.debug('publish %r (%d bytes)', topic, len(raw))

    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None:
        """Send a query and block until a reply arrives or timeout elapses."""
        event = threading.Event()
        result: list[bytes | None] = [None]

        if self._is_advertised(topic):

            def reply_fn(b: bytes) -> None:
                result[0] = b
                event.set()

            self._queue.put((topic, raw, reply_fn))

        _ = event.wait(timeout)
        if result[0] is None:
            logger.warning('query %r timed out after %.1fs', topic, timeout)
        else:
            logger.debug('query %r -> %d bytes', topic, len(result[0]))
        return result[0]

    # ---------------------------- internal helpers ----------------------------

    def _is_advertised(self, topic: str) -> bool:
        return any(matches(pattern, topic) for pattern in self._advertisers)

    def _receiver_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is SENTINEL:
                break
            assert isinstance(item, tuple)
            topic, raw, reply_fn = item
            if self._on_message is not None:
                self._on_message(topic, raw, reply_fn)
