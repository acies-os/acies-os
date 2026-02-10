"""
Message types for async channel (pub/sub) communication.

This module provides the AciesMsg class for async channel messages.
Sync channel (request/response) uses plain dicts via JSON.

Supports multiple serialization formats:
- JSON (default, human-readable, widely compatible)
- Pickle (Python-native, supports more types)
- MessagePack (compact binary, fast)
"""

import json
from datetime import datetime
from typing import Any, Literal

from acies.core.serde import JsonSerializer, PickleSerializer, Serializer

# Type alias for message types
MessageType = Literal['command', 'heartbeat', 'data']

# ============================================================
# Serializer Registry
# ============================================================

# Registry of available serializers by format ID
_serializers: dict[bytes, Serializer] = {
    b'J': JsonSerializer(),
    b'P': PickleSerializer(),
    # b'M': MsgPackSerializer(),
}

# Default serializer (can be changed globally)
_default_serializer: Serializer = JsonSerializer()


# ============================================================
# Message Class
# ============================================================


class AciesMsg:
    """Message for async channel (pub/sub) communication.

    Message types:
    - 'command': One-way commands (topic subscription, service control, etc.)
    - 'heartbeat': Service liveness/diagnostics
    - 'data': Application data (sensor readings, inference results, etc.)

    Serialization formats:
    - JSON (default): Human-readable, widely compatible
    - Pickle: Python-native, supports more types
    - MessagePack: Compact binary, fast

    Note: The sync channel (request/response via Zenoh queryable) uses
    plain dicts, not AciesMsg objects.

    Example:
        # Data message
        msg = AciesMsg(
            msg_type='data',
            payload=[1, 2, 3, 4],
            reply_to='sensor/geo/ctl',
            metadata={'channel': 'EH3'}
        )

        # Serialize with different formats
        json_bytes = msg.to_bytes()  # Default: JSON
        pickle_bytes = msg.to_bytes(PickleSerializer())
        msgpack_bytes = msg.to_bytes(MsgPackSerializer())

        # Deserialize (auto-detects format)
        restored = AciesMsg.from_bytes(json_bytes)
    """

    def __init__(
        self,
        msg_type: MessageType,
        payload: Any,
        *,
        reply_to: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ):
        """Create a message for the async channel.

        Args:
            msg_type: Message type ('command', 'heartbeat', 'data')
            payload: Message payload (any serializable data)
            reply_to: Topic for replies/identification
            metadata: Optional metadata dict
            timestamp: Message timestamp (defaults to now)

        Example:
            msg = AciesMsg(
                msg_type='data',
                payload={'result': 0.95},
                reply_to='classifier/ctl',
                metadata={'model': 'ResNet50'}
            )
        """
        # Read-only fields (internal)
        self._msg_type: MessageType = msg_type
        self._reply_to: str = reply_to
        self._timestamp: datetime = timestamp if timestamp is not None else datetime.now()

        # Mutable fields (internal, accessed via properties)
        self._payload: Any = payload
        self._metadata: dict[str, Any] = metadata if metadata is not None else {}

    # ============================================================
    # Read-only Properties
    # ============================================================

    @property
    def msg_type(self) -> str:
        """Message type ('command', 'heartbeat', or 'data')."""
        return self._msg_type

    @property
    def reply_to(self) -> str:
        """Reply-to topic."""
        return self._reply_to

    @property
    def timestamp(self) -> datetime:
        """Message timestamp as datetime object."""
        return self._timestamp

    @property
    def timestamp_ns(self) -> int:
        """Timestamp in nanoseconds since epoch (for backward compatibility)."""
        return int(self._timestamp.timestamp() * 1e9)

    # ============================================================
    # Mutable Properties
    # ============================================================

    @property
    def payload(self) -> Any:
        """Message payload (can be dict, list, or any serializable data).

        Returns a reference to the internal payload, so mutations affect the message.
        """
        return self._payload

    @payload.setter
    def payload(self, value: Any) -> None:
        """Set message payload."""
        self._payload = value

    @property
    def metadata(self) -> dict[str, Any]:
        """Message metadata (always returns a dict).

        Returns a reference to the internal metadata dict, so mutations affect the message.
        You can modify it in place: msg.metadata['key'] = 'value'
        """
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        """Set message metadata."""
        self._metadata = value

    # ============================================================
    # Serialization Methods
    # ============================================================

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary.

        Returns:
            dict: Dictionary with keys 'msg_type', 'payload', 'reply_to',
                  'metadata', and 'timestamp' (in nanoseconds).

        Example:
            >>> msg.to_dict()
            {
                'msg_type': 'data',
                'payload': [1, 2, 3],
                'reply_to': 'sensor/ctl',
                'metadata': {'channel': 'EH3'},
                'timestamp': 1234567890000000000
            }
        """
        return {
            'msg_type': self._msg_type,
            'payload': self._payload,
            'reply_to': self._reply_to,
            'metadata': self._metadata,
            'timestamp': self.timestamp_ns,
        }

    def to_json(self) -> str:
        """Convert message to JSON string.

        Returns:
            str: JSON-encoded message.

        Example:
            >>> msg.to_json()
            '{"msg_type": "data", "payload": [1, 2, 3], ...}'
        """
        return json.dumps(self.to_dict())

    def to_bytes(self, serializer: Serializer | None = None) -> bytes:
        """Serialize message to bytes.

        The serialized format includes a 1-byte format marker for auto-detection
        during deserialization.

        Args:
            serializer: Serializer to use (defaults to global default, usually JSON)

        Returns:
            bytes: Serialized message with format marker prefix

        Example:
            >>> # Use default serializer (JSON)
            >>> data = msg.to_bytes()
            >>>
            >>> # Use specific serializer
            >>> data = msg.to_bytes(MsgPackSerializer())
            >>>
            >>> # Deserialize (auto-detects format)
            >>> restored = AciesMsg.from_bytes(data)
        """
        ser = serializer or _default_serializer
        data = ser.serialize(self.to_dict())
        return ser.format_id + data

    @classmethod
    def from_bytes(cls, data: bytes) -> 'AciesMsg':
        """Deserialize message from bytes (auto-detects format).

        Args:
            data: Serialized message bytes (with format marker)

        Returns:
            AciesMsg: Deserialized message instance

        Raises:
            ValueError: If data is empty or format is unknown
            Various: Deserialization errors depending on format

        Example:
            >>> data = msg.to_bytes()
            >>> restored = AciesMsg.from_bytes(data)
        """
        if len(data) < 1:
            raise ValueError('Cannot deserialize empty data')

        # Detect format from marker
        format_id = data[0:1]
        payload = data[1:]

        if format_id not in _serializers:
            raise ValueError(f'Unknown serialization format: {format_id!r}. Known formats: {list(_serializers.keys())}')

        serializer = _serializers[format_id]
        return cls.from_dict(serializer.deserialize(payload))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'AciesMsg':
        """Create message from dictionary.

        Args:
            d: Dictionary with message fields

        Returns:
            AciesMsg: Message instance

        Example:
            >>> msg = AciesMsg.from_dict({
            ...     'msg_type': 'data',
            ...     'payload': [1, 2, 3],
            ...     'reply_to': 'sensor/ctl',
            ...     'metadata': {},
            ...     'timestamp': 1234567890000000000
            ... })
        """
        return cls(
            msg_type=d['msg_type'],
            payload=d['payload'],
            reply_to=d['reply_to'],
            metadata=d.get('metadata', {}),
            timestamp=datetime.fromtimestamp(d['timestamp'] / 1e9),
        )

    # ============================================================
    # Class Methods for Serializer Management
    # ============================================================

    @classmethod
    def register_serializer(cls, serializer: Serializer) -> None:
        """Register a custom serializer.

        Args:
            serializer: Serializer instance to register

        Raises:
            ValueError: If format_id conflicts with existing serializer

        Example:
            >>> class ZstdJsonSerializer:
            ...     format_id = b'Z'
            ...     def serialize(self, data): ...
            ...     def deserialize(self, data): ...
            >>>
            >>> AciesMsg.register_serializer(ZstdJsonSerializer())
        """
        if serializer.format_id in _serializers:
            existing = _serializers[serializer.format_id]
            if type(existing) != type(serializer):
                raise ValueError(f'Format ID {serializer.format_id!r} already registered by {type(existing).__name__}')
        _serializers[serializer.format_id] = serializer

    @classmethod
    def set_default_serializer(cls, serializer: Serializer) -> None:
        """Set the default serializer for new messages.

        Args:
            serializer: Serializer instance to use as default

        Example:
            >>> # Use MessagePack by default
            >>> AciesMsg.set_default_serializer(MsgPackSerializer())
            >>>
            >>> # Now all messages use msgpack unless specified
            >>> data = msg.to_bytes()  # Uses msgpack
        """
        global _default_serializer
        _default_serializer = serializer

    @classmethod
    def get_default_serializer(cls) -> Serializer:
        """Get the current default serializer.

        Returns:
            Serializer: Current default serializer instance
        """
        return _default_serializer

    @classmethod
    def list_serializers(cls) -> dict[bytes, str]:
        """List all registered serializers.

        Returns:
            dict: Mapping of format_id to serializer class name

        Example:
            >>> AciesMsg.list_serializers()
            {b'J': 'JsonSerializer', b'P': 'PickleSerializer', b'M': 'MsgPackSerializer'}
        """
        return {fmt: type(ser).__name__ for fmt, ser in _serializers.items()}

    # ============================================================
    # Utility Methods
    # ============================================================

    def copy(self) -> 'AciesMsg':
        """Create a deep copy of this message.

        Returns:
            AciesMsg: New message instance with same data

        Example:
            >>> msg_copy = msg.copy()
            >>> msg_copy.metadata['new_key'] = 'value'  # Doesn't affect original
        """
        return self.from_dict(self.to_dict())

    def __repr__(self) -> str:
        """String representation of message.

        Returns:
            str: Human-readable representation

        Example:
            >>> msg
            AciesMsg(msg_type='data', reply_to='sensor/ctl', timestamp=2024-01-15 10:30:45)
        """
        return (
            f"AciesMsg(msg_type='{self._msg_type}', "
            f"reply_to='{self._reply_to}', "
            f'timestamp={self._timestamp.strftime("%Y-%m-%d %H:%M:%S")})'
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another message.

        Args:
            other: Object to compare with

        Returns:
            bool: True if messages are equal
        """
        if not isinstance(other, AciesMsg):
            return False
        return self.to_dict() == other.to_dict()
