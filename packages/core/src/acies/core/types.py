"""
Message types for async channel (pub/sub) communication.

This module provides the AciesMsg class for async channel messages.
Sync channel (request/response) uses plain dicts via JSON.
"""

import json
from datetime import datetime
from typing import Any, Literal


class AciesMsg:
    """Message for async channel (pub/sub) communication.

    Message types:
    - 'command': One-way commands (topic subscription, service control, etc.)
    - 'heartbeat': Service liveness/diagnostics
    - 'data': Application data (sensor readings, inference results, etc.)

    Note: The sync channel (request/response via Zenoh queryable) uses
    plain dicts, not AciesMsg objects.

    Example:
        # Data message
        msg = AciesMsg(
            type='data',
            payload=[1, 2, 3, 4],
            reply_to='sensor/geo/ctl',
            metadata={'channel': 'EH3'}
        )

        # Heartbeat
        msg = AciesMsg(
            type='heartbeat',
            payload={'cpu': 0.5, 'mem': 0.3},
            reply_to='service/ctl',
            metadata={'deactivated': False}
        )

        # Command
        msg = AciesMsg(
            type='command',
            payload={'command': 'subscribe', 'topic': 'sensors/geo'},
            reply_to='service/ctl'
        )
    """

    def __init__(
        self,
        msg_type: Literal['command', 'heartbeat', 'data'],
        payload: Any,
        *,
        reply_to: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ):
        """Create a message for the async channel.

        Args:
            type: Message type ('command', 'heartbeat', 'data')
            payload: Message payload (any JSON-serializable data)
            reply_to: Topic for replies/identification
            metadata: Optional metadata dict
            timestamp: Message timestamp (defaults to now)

        Example:
            msg = AciesMsg(
                type='data',
                payload={'result': 0.95},
                reply_to='classifier/ctl',
                metadata={'model': 'ResNet50'}
            )
        """
        self._type = msg_type
        self._payload = payload
        self._reply_to = reply_to
        self._metadata = metadata if metadata is not None else {}
        self._timestamp = timestamp if timestamp is not None else datetime.now()

    @property
    def type(self) -> str:
        """Message type ('command', 'heartbeat', or 'data')."""
        return self._type

    @property
    def payload(self) -> Any:
        """Message payload (can be dict, list, or any JSON-serializable data)."""
        return self._payload

    @payload.setter
    def payload(self, value: Any):
        """Set message payload."""
        self._payload = value

    @property
    def reply_to(self) -> str:
        """Reply-to topic."""
        return self._reply_to

    @property
    def metadata(self) -> dict:
        """Message metadata (always returns a dict)."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        """Set message metadata."""
        if not isinstance(value, dict):
            raise TypeError(f'Metadata must be a dict, got {type(value)}')
        self._metadata = value

    @property
    def timestamp(self) -> datetime:
        """Message timestamp as datetime object."""
        return self._timestamp

    @property
    def timestamp_ns(self) -> int:
        """Timestamp in nanoseconds since epoch (for backward compatibility)."""
        return int(self._timestamp.timestamp() * 1e9)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary.

        Returns:
            dict: Dictionary with keys 'type', 'payload', 'reply_to',
                  'metadata', and 'timestamp' (in nanoseconds).

        Example:
            >>> msg.to_dict()
            {
                'type': 'data',
                'payload': [1, 2, 3],
                'reply_to': 'sensor/ctl',
                'metadata': {'channel': 'EH3'},
                'timestamp': 1234567890000000000
            }
        """
        return {
            'type': self._type,
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
            '{"type": "data", "payload": [1, 2, 3], ...}'
        """
        return json.dumps(self.to_dict())

    def to_bytes(self) -> bytes:
        """Serialize message to bytes (JSON-encoded UTF-8).

        Returns:
            bytes: Serialized message suitable for transport.

        Example:
            >>> data = msg.to_bytes()
            >>> restored = AciesMsg.from_bytes(data)
        """
        return self.to_json().encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> 'AciesMsg':
        """Deserialize message from bytes.

        Args:
            data: Serialized message bytes (JSON-encoded UTF-8)

        Returns:
            AciesMsg: Deserialized message instance

        Raises:
            json.JSONDecodeError: If data is not valid JSON
            UnicodeDecodeError: If data is not valid UTF-8

        Example:
            >>> data = msg.to_bytes()
            >>> restored = AciesMsg.from_bytes(data)
        """
        d = json.loads(data.decode('utf-8'))
        return cls(
            type=d['type'],
            payload=d['payload'],
            reply_to=d['reply_to'],
            metadata=d.get('metadata', {}),
            timestamp=datetime.fromtimestamp(d['timestamp'] / 1e9),
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'AciesMsg':
        """Create message from dictionary.

        Args:
            d: Dictionary with message fields

        Returns:
            AciesMsg: Message instance

        Example:
            >>> msg = AciesMsg.from_dict({
            ...     'type': 'data',
            ...     'payload': [1, 2, 3],
            ...     'reply_to': 'sensor/ctl',
            ...     'metadata': {},
            ...     'timestamp': 1234567890000000000
            ... })
        """
        return cls(
            type=d['type'],
            payload=d['payload'],
            reply_to=d['reply_to'],
            metadata=d.get('metadata', {}),
            timestamp=datetime.fromtimestamp(d['timestamp'] / 1e9),
        )

    def copy(self) -> 'AciesMsg':
        """Create a deep copy of this message.

        Returns:
            AciesMsg: New message instance with same data

        Example:
            >>> msg_copy = msg.copy()
            >>> msg_copy.metadata['new_key'] = 'value'  # Doesn't affect original
        """
        # Use to_dict/from_dict for deep copy via serialization
        return self.from_dict(self.to_dict())

    def __repr__(self) -> str:
        """String representation of message.

        Returns:
            str: Human-readable representation

        Example:
            >>> msg
            AciesMsg(type='data', reply_to='sensor/ctl', timestamp=2024-01-15 10:30:45)
        """
        return (
            f"AciesMsg(type='{self._type}', "
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


# ============================================================
# Backward Compatibility Helpers
# ============================================================
# These functions help migrate from the old API to the new one


def get_payload(msg: AciesMsg) -> Any:
    """Get message payload (backward compatibility helper).

    Args:
        msg: Message instance

    Returns:
        Payload data

    Note:
        This is a backward compatibility helper.
        New code should use msg.payload directly.
    """
    return msg.payload


def get_metadata(msg: AciesMsg) -> dict:
    """Get message metadata (backward compatibility helper).

    Args:
        msg: Message instance

    Returns:
        Metadata dict

    Note:
        This is a backward compatibility helper.
        New code should use msg.metadata directly.
    """
    return msg.metadata


def set_metadata(msg: AciesMsg, value: dict):
    """Set message metadata (backward compatibility helper).

    Args:
        msg: Message instance
        value: Metadata dict to set

    Note:
        This is a backward compatibility helper.
        New code should use msg.metadata = value directly.
    """
    msg.metadata = value
