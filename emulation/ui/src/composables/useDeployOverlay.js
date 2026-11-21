// The configuring window: after a deploy the testbed gets about 30 s (the gateway's
// _REPLAY_AHEAD_TIME_S) to configure the nodes and synchronize their start (start_at).
// The window counts down to that start, for deploys from this page or the main UI alike.
import { ref, watch } from 'vue';
import { useMonitor } from './useMonitor.js';

const visible = ref(false);
const progress = ref(0); // 0..100
const message = ref('');
let sentAt = null; // when this page sent a deploy (ms)
let countingTo = null; // the start_at being counted down to
let dismissed = false;
let started = false;

export function useDeployOverlay() {
  const mon = useMonitor();

  if (!started) {
    started = true;
    watch(mon.snap, update);
    mon.onAck((ack) => {
      if (!ack.ok) {
        sentAt = null;
        visible.value = false;
      }
    });
  }

  function update(s) {
    if (!s) return;
    const t = s.t_replay;
    if (t !== null && t < 0) {
      if (s.start_at !== countingTo) { // a new deploy is counting down
        countingTo = s.start_at;
        dismissed = false;
        if (!sentAt || Date.now() - sentAt > 20000) sentAt = Date.now();
      }
      progress.value = Math.min(100, (100 * (Date.now() - sentAt)) / Math.max(1, s.start_at * 1000 - sentAt));
      message.value = `Configuring the nodes and synchronizing their start time. Replay starts in ${Math.ceil(-t)} s.`;
      visible.value = !dismissed;
    } else if (sentAt && Date.now() - sentAt < 20000 && s.start_at === countingTo) {
      progress.value = 3; // sent; the emulator has not armed the new start yet
      message.value = 'Sending the configuration to the nodes…';
      visible.value = !dismissed;
    } else {
      sentAt = null;
      visible.value = false;
    }
  }

  return {
    visible, progress, message,
    begin() {
      sentAt = Date.now();
      dismissed = false;
      progress.value = 3;
      message.value = 'Sending the configuration to the nodes…';
      visible.value = true;
    },
    dismiss() {
      dismissed = true;
      visible.value = false;
    },
  };
}
