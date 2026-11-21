// Link state as shown in the table and on the map.

/**
 * State of one link from its emu/state entry.
 * @returns {{label: string, cls: 'up'|'impaired'|'down'|'unknown'}}
 */
export function linkStatus(link) {
  if (!link || !link.config) return { label: '–', cls: 'unknown' };
  if (link.unplugged) return { label: 'Unplugged', cls: 'down' };
  if (link.down) return { label: 'Down', cls: 'down' };
  const c = link.config;
  const impaired = (c.loss || 0) > 0 || parseFloat(c.delay || 0) >= 50 || c.jitter || (c.bw !== undefined && c.bw < 1);
  return impaired ? { label: 'Impaired', cls: 'impaired' } : { label: 'Up', cls: 'up' };
}

/** "5ms ±2ms", or null when no delay is set. */
export function delayText(config = {}) {
  if (!config.delay) return null;
  return config.jitter ? `${config.delay} ±${config.jitter}` : config.delay;
}
