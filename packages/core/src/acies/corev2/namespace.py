import re


def matches(pattern: str, topic: str) -> bool:
    """Return True if topic matches pattern.

    Supports zenoh-style wildcards:
      *   — exactly one chunk (non-empty sequence of non-'/' chars)
      **  — any number of chunks, including zero (may span multiple '/' separators)
    Exact match always works.
    """
    if pattern == topic:
        return True
    regex = ''
    i = 0
    while i < len(pattern):
        if pattern[i : i + 2] == '**':
            regex += '.*'
            i += 2
        elif pattern[i] == '*':
            regex += '[^/]+'
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return bool(re.fullmatch(regex, topic))
