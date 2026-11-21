// The fault plan: rows edited on the page <-> emulator schedule events.
//
// A row is {at, nodes, change, value, dur}: at `at` seconds after the replay starts, apply
// `change` to `nodes`; with a duration, revert it `dur` seconds later. An event is what
// topo.py runs: {at, nodes, action, bw/delay/jitter/loss}.

export const CHANGES = {
  down: { label: 'Blackout', unit: null, revert: 'up' },
  unplug: { label: 'Unplug', unit: null, revert: 'plug' },
  loss: { label: 'Loss', unit: '%', revert: 'reset' },
  delay: { label: 'Delay', unit: 'ms', revert: 'reset' },
  jitter: { label: 'Jitter', unit: 'ms', revert: 'reset' },
  bw: { label: 'Bandwidth', unit: 'Mbit/s', revert: 'reset' },
  up: { label: 'End blackout', unit: null },
  plug: { label: 'Plug back in', unit: null },
  reset: { label: 'Restore link', unit: null },
};
const SET_KEYS = ['loss', 'delay', 'jitter', 'bw'];

export const newRow = (at = 30) => ({ at, nodes: [], change: 'down', value: '', dur: 30 });

/** Common plans, given the emulated node names. */
export function presets(nodes) {
  return {
    blackout: [{ at: 40, nodes: nodes.slice(2, 5), change: 'down', value: '', dur: 40 }],
    loss: [{ at: 40, nodes: [...nodes], change: 'loss', value: 20, dur: 40 }],
    edge: [{ at: 40, nodes: ['edge'], change: 'bw', value: 1, dur: 40 }],
  };
}

function expandNodes(spec, allNodes) {
  if (spec === 'all') return [...allNodes];
  if (spec === 'all+edge') return [...allNodes, 'edge'];
  return [].concat(spec ?? []);
}

/** Rows -> events; a row with a duration becomes the change plus its revert. */
export function eventsFromRows(rows) {
  const events = [];
  for (const r of rows) {
    if (!r.nodes.length || r.at === '' || Number.isNaN(Number(r.at))) continue;
    const kind = CHANGES[r.change];
    const ev = { at: Number(r.at), nodes: r.nodes };
    if (kind.unit) {
      if (r.value === '' || Number.isNaN(Number(r.value))) continue;
      ev.action = 'set';
      ev[r.change] = kind.unit === 'ms' ? `${Number(r.value)}ms` : Number(r.value);
    } else {
      ev.action = r.change;
    }
    events.push(ev);
    if (kind.revert && r.dur !== '' && Number(r.dur) > 0) {
      events.push({ at: Number(r.at) + Number(r.dur), nodes: r.nodes, action: kind.revert });
    }
  }
  return events.sort((a, b) => a.at - b.at);
}

/** Events -> rows; a change followed by its revert on the same links folds into one row. */
export function rowsFromEvents(events, allNodes) {
  const rows = [];
  const open = []; // rows a later up/plug/reset can close
  const key = (nodes) => [...nodes].sort().join(',');
  const closes = { up: ['down'], plug: ['unplug'], reset: SET_KEYS };
  for (const e of events) {
    const nodes = expandNodes(e.nodes ?? e.node, allNodes);
    const closable = closes[e.action];
    if (closable) {
      const matches = open.filter((r) => closable.includes(r.change) && key(r.nodes) === key(nodes));
      if (matches.length) {
        for (const r of matches) {
          r.dur = e.at - r.at;
          open.splice(open.indexOf(r), 1);
        }
        continue;
      }
    }
    if (e.action === 'set') {
      for (const k of SET_KEYS) {
        if (e[k] === undefined) continue;
        const r = { at: e.at, nodes, change: k, value: parseFloat(e[k]), dur: '' };
        rows.push(r);
        open.push(r);
      }
    } else {
      const r = { at: e.at, nodes, change: e.action, value: '', dur: '' };
      rows.push(r);
      if (CHANGES[e.action]?.revert) open.push(r);
    }
  }
  return rows;
}
