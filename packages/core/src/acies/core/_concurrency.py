from typing import final


@final
class Sentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return 'STOP_SIGNAL'


SENTINEL = Sentinel()
