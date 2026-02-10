import json
import pickle
from typing import Any, Protocol

# ============================================================
# Serializer Protocol and Implementations
# ============================================================


class Serializer(Protocol):
    """Protocol for message serializers.

    Custom serializers must implement this protocol.
    """

    format_id: bytes  # Single-byte format identifier

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize dict to bytes."""
        ...

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize bytes to dict."""
        ...


class JsonSerializer:
    """JSON serializer (default).

    Pros: Human-readable, widely compatible, language-agnostic
    Cons: Larger size, slower than binary formats
    """

    format_id: bytes = b'J'

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize dict to JSON bytes."""
        return json.dumps(data).encode('utf-8')

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize JSON bytes to dict."""
        return json.loads(data.decode('utf-8'))


class PickleSerializer:
    """Pickle serializer.

    Pros: Python-native, supports more types (datetime, custom classes, etc.)
    Cons: Python-only, security risk with untrusted data, version-dependent

    Warning: Only use pickle with trusted data sources!
    """

    format_id: bytes = b'P'

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize dict to pickle bytes."""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize pickle bytes to dict."""
        return pickle.loads(data)


# class MsgPackSerializer:
#     """MessagePack serializer.

#     Pros: Compact binary format, faster than JSON, language-agnostic
#     Cons: Requires msgpack library
#     """

#     format_id: bytes = b'M'

#     def serialize(self, data: dict[str, Any]) -> bytes:
#         """Serialize dict to msgpack bytes."""
#         try:
#             import msgpack
#         except ImportError as e:
#             raise ImportError(
#                 "MessagePack serializer requires 'msgpack' library. Install it with: pip install msgpack"
#             ) from e

#         return msgpack.packb(data, use_bin_type=True)

#     def deserialize(self, data: bytes) -> dict[str, Any]:
#         """Deserialize msgpack bytes to dict."""
#         try:
#             import msgpack
#         except ImportError as e:
#             raise ImportError(
#                 "MessagePack serializer requires 'msgpack' library. Install it with: pip install msgpack"
#             ) from e

#         return msgpack.unpackb(data, raw=False, strict_map_key=False)
