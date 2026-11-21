// Per-browser conveniences (last plan, last target). Storage can be unavailable (private
// windows, blocked site data), so every access is guarded and the app works without it.
const PREFIX = 'acies-emu-';

export function load(key, fallback = null) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function save(key, value) {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    /* storage unavailable */
  }
}
