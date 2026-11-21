// Connection to the monitor (emulation/monitor.py) and the state it sends.
//
// Messages from the monitor (JSON over /ws):
//   {type: 'sites', sites}        once on connect: every site config (nodes, road, targets)
//   {type: 'snapshot', ...}       once a second: links, SPAR, schedule, events (see DESIGN.md)
//   {type: 'ack', ok, message}    reply to a command
// Commands to it: {type: 'fault' | 'deploy' | 'plan', ...}.
//
// One shared store for the whole app: every component calls useMonitor() and gets the same state.
import { computed, reactive, shallowRef } from 'vue';
import { SPAR_STALE_S } from '../config.js';

const state = reactive({
  connected: false,
  sites: {},
  toast: { text: '', error: false },
});
const snap = shallowRef(null); // replaced as a whole every second
const ackListeners = new Set();
let socket = null;
let started = false;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${proto}://${location.host}/ws`);
  socket.onopen = () => { state.connected = true; };
  socket.onclose = () => {
    state.connected = false;
    setTimeout(connect, 2000);
  };
  socket.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'sites') state.sites = msg.sites;
    else if (msg.type === 'snapshot') snap.value = msg;
    else if (msg.type === 'ack') {
      toast(msg.message, !msg.ok);
      ackListeners.forEach((fn) => fn(msg));
    }
  };
}

function toast(text, error = false) {
  state.toast = { text, error };
}

function send(cmd) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    toast('not connected to the monitor', true);
    return false;
  }
  socket.send(JSON.stringify(cmd));
  return true;
}

/** Change links now, e.g. fault('down', ['rs3']) or fault('set', 'all', {loss: 20}). */
const fault = (action, nodes, params = {}) => send({ type: 'fault', action, nodes, ...params });

const nodes = computed(() => snap.value?.nodes || []);
const links = computed(() => snap.value?.links || {});
const tReplay = computed(() => snap.value?.t_replay ?? null);
const windows = computed(() => (snap.value?.windows || []).filter((w) => w.t !== null));
// SPAR is running while its newest window is recent (not before a deploy or after the replay)
const sparRunning = computed(() => {
  const newest = windows.value[windows.value.length - 1];
  return !!newest && tReplay.value !== null && tReplay.value - newest.t <= SPAR_STALE_S;
});
// link changes of the current replay, in seconds after its start
const replayEvents = computed(() => {
  const s = snap.value;
  if (!s?.start_at) return [];
  return (s.events || [])
    .filter((e) => !['armed', 'plan'].includes(e.action) && e.start_at === s.start_at)
    .map((e) => ({ ...e, t: e.wall_ns / 1e9 - s.start_at }))
    .filter((e) => e.t >= 0);
});

export function useMonitor() {
  if (!started) {
    started = true;
    connect();
  }
  return {
    state, snap, nodes, links, tReplay, windows, sparRunning, replayEvents,
    send, fault, toast,
    onAck: (fn) => { ackListeners.add(fn); return () => ackListeners.delete(fn); },
  };
}
