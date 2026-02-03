import json
from typing import Any

from ._acies_core import Msg, from_bytes, to_bytes


def _encode(data: Any | bytes) -> bytes:
    if isinstance(data, bytes):
        return data
    elif isinstance(data, str):
        return data.encode()
    elif data is None:
        return b''
    else:
        return json.dumps(data).encode()


def _decode(data: bytes) -> Any:
    if len(data) == 0:
        return None
    else:
        return json.loads(data)


class Message:
    def __init__(
        self,
        msg_type: str,
        timestamp: int,
        reply_to: str,
        payload: bytes | Any,
        metadata: bytes | Any,
    ):
        payload = _encode(payload)
        metadata = _encode(metadata)
        self._msg = Msg(msg_type, timestamp, reply_to, payload, metadata)

    @property
    def msg_type(self) -> str:
        return self._msg.msg_type()

    @property
    def timestamp(self) -> int:
        return self._msg.timestamp()

    @property
    def reply_to(self) -> str:
        return self._msg.reply_to()

    def get_payload(self) -> Any:
        data = self._msg.payload()
        try:
            data = _decode(data)
        except Exception:
            pass
        return data

    def get_metadata(self) -> Any:
        data = self._msg.metadata()
        try:
            data = _decode(data)
        except Exception:
            pass
        return data

    def to_bytes(self) -> bytes:
        return to_bytes(self._msg)

    @classmethod
    def from_bytes(cls, data: bytes):
        obj = cls.__new__(cls)
        super(Message, obj).__init__()
        obj._msg = from_bytes(data)
        return obj

    def to_dict(self) -> dict:
        payload = _decode(self._msg.payload())
        metadata = _decode(self._msg.metadata())
        return {
            'msg_type': self.msg_type,
            'timestamp': self.timestamp,
            'reply_to': self.reply_to,
            'payload': payload,
            'metadata': metadata,
        }

    def to_json(self) -> str:
        data = self.to_dict()
        # convert bytes to string so that it can be serialized in JSON
        if isinstance(data['payload'], bytes):
            data['payload'] = data['payload'].decode()
        if isinstance(data['metadata'], bytes):
            data['metadata'] = data['metadata'].decode()
        return json.dumps(data)

    def __repr__(self) -> str:
        return f'Message(msg_type="{self.msg_type}", timestamp={self.timestamp}, reply_to="{self.reply_to}"), payload={self.get_payload()}, metadata={self.get_metadata()})'
