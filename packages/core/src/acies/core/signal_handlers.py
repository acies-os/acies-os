import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Callable


@contextmanager
def temporary_signal_handlers(stop: Callable[..., None]) -> Iterator[None]:
    def _handle_signal(signum: int, frame: FrameType | None) -> None:  # pyright: ignore[reportUnusedParameter]
        stop()

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)

    _ = signal.signal(signal.SIGINT, _handle_signal)
    _ = signal.signal(signal.SIGTERM, _handle_signal)
    try:
        yield
    finally:
        _ = signal.signal(signal.SIGINT, old_sigint)
        _ = signal.signal(signal.SIGTERM, old_sigterm)
