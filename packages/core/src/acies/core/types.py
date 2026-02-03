import json

from ._acies_core import _Msg


def _dumps(data: dict | None) -> bytes:
    """Serialize a mapping to UTF-8 JSON bytes.

    Args:
        data: JSON-serializable mapping or ``None``.

    Returns:
        bytes: UTF-8 encoded JSON (``b"null"`` if ``None``).
    """

    return json.dumps(data).encode()


def _loads(data: bytes) -> dict:
    """Deserialize UTF-8 JSON bytes to a Python object.

    Args:
        data: UTF-8 JSON payload.

    Returns:
        dict: Decoded mapping.

    Raises:
        json.JSONDecodeError: If the payload is not valid JSON.
        UnicodeDecodeError: If the bytes are not UTF-8.
    """
    return json.loads(data)


_new_array_constructors = {
    'i16': _Msg.new_array_i16,
    'i32': _Msg.new_array_i32,
    'i64': _Msg.new_array_i64,
    'f64': _Msg.new_array_f64,
}


class AciesMsg:
    """High-level wrapper around the C-extension ``_Msg``.

    Provides constructors for array, json, heartbeat, and control messages,
    plus helpers to access payload/metadata as Python objects and to
    convert to/from raw bytes.

    Notes:
        * Timestamps are nanoseconds since epoch.
        * Payloads/metadata are stored as UTF-8 JSON when applicable.
    """

    def __init__(self):
        self._msg = None

    @classmethod
    def new_array_msg(
        cls,
        payload: list[int | float],
        reply_to: str,
        metadata: dict | None,
        data_type='i16',
        timestamp_ns: int | None = None,
    ):
        """Create an array message.

        Args:
            payload: Numeric values to send.
            reply_to: Topic to address replies to.
            metadata: Optional mapping attached as metadata.
            data_type: One of ``{"i16","i32","i64","f64"}``.
            timestamp_ns: Optional nanosecond timestamp to set.

        Returns:
            AciesMsg: The constructed message.

        Raises:
            ValueError: If ``data_type`` is not supported.
        """

        if data_type not in _new_array_constructors:
            raise ValueError(f'Invalid data type {data_type}, currently only i16, i32, i64, f64 are supported.')
        func = _new_array_constructors[data_type]
        _metadata = _dumps(metadata)
        obj = cls.__new__(cls)
        super(AciesMsg, obj).__init__()
        obj._msg = func(payload, reply_to, _metadata)
        if timestamp_ns is not None:
            obj._msg.set_timestamp(timestamp_ns)
        return obj

    @classmethod
    def new_heartbeat(
        cls,
        reply_to: str,
        metadata: dict | None,
        timestamp_ns: int | None = None,
    ):
        """Create a heartbeat message.

        Args:
            reply_to: Topic to address replies to.
            metadata: Optional mapping attached as metadata.
            timestamp_ns: Optional nanosecond timestamp to set.

        Returns:
            AciesMsg: The heartbeat message.
        """

        _metadata = _dumps(metadata)
        obj = cls.__new__(cls)
        super(AciesMsg, obj).__init__()
        obj._msg = _Msg.new_heartbeat(reply_to, _metadata)
        if timestamp_ns is not None:
            obj._msg.set_timestamp(timestamp_ns)
        return obj

    @classmethod
    def new_ctl_msg(
        cls,
        kind: str,
        reply_to: str,
        payload: dict,
        metadata: dict | None = None,
        timestamp_ns: int | None = None,
    ):
        """Create a control message.

        Args:
            kind: One of ``{"set","get","topic","reply"}``.
            reply_to: Control topic.
            payload: Control payload mapping (JSON-serializable).
            metadata: Optional metadata mapping.
            timestamp_ns: Optional nanosecond timestamp to set.

        Returns:
            AciesMsg: The control message.

        Raises:
            ValueError: If ``kind`` is not supported.
        """
        if kind not in ['set', 'get', 'topic', 'reply']:
            raise ValueError(f'Invalid kind: {kind}')
        _payload = _dumps(payload)
        _metadata = _dumps(metadata)
        obj = cls.__new__(cls)
        super(AciesMsg, obj).__init__()
        obj._msg = _Msg.new_ctl(kind, reply_to, _payload, _metadata)
        if timestamp_ns is not None:
            obj._msg.set_timestamp(timestamp_ns)
        return obj

    @classmethod
    def new_json_msg(
        cls,
        reply_to: str,
        payload: dict,
        metadata: dict | None = None,
        timestamp_ns: int | None = None,
    ):
        """Create a JSON message.

        Args:
            reply_to: Topic to address replies to.
            payload: JSON-serializable mapping as the message body.
            metadata: Optional metadata mapping.
            timestamp_ns: Optional nanosecond timestamp to set.

        Returns:
            AciesMsg: The JSON message.
        """

        _payload = _dumps(payload)
        _metadata = _dumps(metadata)
        obj = cls.__new__(cls)
        super(AciesMsg, obj).__init__()
        obj._msg = _Msg.new_json(reply_to, _payload, _metadata)
        if timestamp_ns is not None:
            obj._msg.set_timestamp(timestamp_ns)
        return obj

    @property
    def timestamp(self) -> int:
        assert self._msg is not None
        return self._msg.timestamp

    @property
    def reply_to(self) -> str:
        assert self._msg is not None
        return self._msg.reply_to

    def get_payload(self) -> dict:
        """Return the decoded payload.

        Returns:
            dict: Decoded payload mapping (empty dict if ``None``).

        """
        assert self._msg is not None
        data = self._msg.payload
        if isinstance(data, bytes):
            data = _loads(data)
        if data is None:
            data = {}
        return data

    def get_metadata(self) -> dict:
        """Return the decoded metadata.

        Returns:
            dict: Decoded metadata mapping (empty dict if ``None``).
        """
        
        assert self._msg is not None
        data = self._msg.metadata
        if isinstance(data, bytes):
            data = _loads(data)
        if data is None:
            data = {}
        return data

    def set_metadata(self, value: dict):
        """Set metadata for message kinds that support it.

        Supported kinds: ``array_i16``, ``array_i32``, ``array_i64``,
        ``array_f64``, ``set``, ``get``, ``reply``, ``topic``.

        Args:
            value: Mapping to serialize as metadata.

        Raises:
            TypeError: If the current message kind does not support metadata.
        """

        assert self._msg is not None
        if self.kind in [
            'array_i16',
            'array_i32',
            'array_i64',
            'array_f64',
            'set',
            'get',
            'reply',
            'topic',
        ]:
            self._msg.metadata = _dumps(value)
        else:
            raise TypeError(f'{self.kind} msg does not have metadata')

    @property
    def kind(self) -> str:
        """str: Message kind (e.g., ``"json"``, ``"reply"``, ``"array_i16"``)."""

        assert self._msg is not None
        return self._msg.kind

    def __repr__(self):
        return self._msg.__repr__()

    def to_bytes(self):
        """Serialize the message to its wire format.

        Returns:
            bytes: Encoded message suitable for transport/storage.
        """
        assert self._msg is not None
        return self._msg.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes):
        """Create a message from its wire-format bytes.

        Args:
            data: Byte representation produced by :meth:`to_bytes`.

        Returns:
            AciesMsg: Decoded message instance.
        """
        obj = cls.__new__(cls)
        super(AciesMsg, obj).__init__()
        obj._msg = _Msg.from_bytes(data)
        return obj

    def to_dict(self):
        """Return a JSON-friendly dictionary view of the message.

        Returns:
            dict: Mapping with keys ``"kind"``, ``"timestamp"``, ``"reply_to"``,
            ``"payload"``, and ``"metadata"``.
        """
        return {
            'kind': self.kind,
            'timestamp': self.timestamp,
            'reply_to': self.reply_to,
            'payload': self.get_payload(),
            'metadata': self.get_metadata(),
        }

    def to_json(self):
        """Return a JSON string representation of :meth:`to_dict`.

        Returns:
            str: JSON-encoded message summary.
        """
        val = self.to_dict()
        json_str = json.dumps(val)
        return json_str
