<script setup>
// A value per SPAR window over replay time, with a dashed line at each link change.
//   value(w) -> number | null    what to plot for window w (null leaves a gap)
//   mode     'line' (point at each window's start) or 'step' (held for the 2-s window)
//   yMax / yTicks / yFormat      y scale; yMax 'auto' picks a round maximum from the data
//   tooltip(w) -> html           optional hover text
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useMonitor } from '../composables/useMonitor.js';
import { drawAxes, drawMarks, niceStep, PAD, setupCanvas } from '../lib/chart.js';

const props = defineProps({
  value: { type: Function, required: true },
  mode: { type: String, default: 'line' },
  color: { type: String, default: '#e05339' },
  height: { type: Number, default: 220 },
  yMax: { type: [Number, String], default: 'auto' },
  yFormat: { type: Function, default: (v) => `${v}` },
  labels: { type: Boolean, default: true }, // label the link-change lines
  tooltip: { type: Function, default: null },
});

const { windows, tReplay, replayEvents } = useMonitor();
const canvas = ref(null);
const tip = ref({ show: false, html: '', left: 0, top: 0 });
let hoverT = null;
let observer = null;

const WINDOW_S = 2;
const xMax = () => Math.max(120, (tReplay.value || 0) + 5, ...windows.value.map((w) => w.t + WINDOW_S));

function scaleY(values) {
  if (props.yMax !== 'auto') {
    const n = props.yMax;
    return { max: n, ticks: [0, Math.ceil(n / 2), n] };
  }
  const sorted = values.filter((v) => v !== null).sort((a, b) => a - b);
  const step = niceStep((sorted.length ? sorted[Math.floor(0.98 * (sorted.length - 1))] : 20) / 4);
  return { max: 4 * step, ticks: [0, step, 2 * step, 3 * step, 4 * step] };
}

function nearest(t) {
  let best = null;
  for (const w of windows.value) if (!best || Math.abs(w.t - t) < Math.abs(best.t - t)) best = w;
  return best && Math.abs(best.t - t) <= 6 ? best : null;
}

function draw() {
  if (!canvas.value) return;
  const [ctx, w, h] = setupCanvas(canvas.value, props.height);
  const ws = windows.value;
  const vals = ws.map(props.value);
  const { max, ticks } = scaleY(vals);
  const xm = xMax();
  const [X, Y] = drawAxes(ctx, w, h, xm, max, ticks, props.yFormat);
  drawMarks(ctx, X, h, replayEvents.value, tReplay.value || 0, props.labels);

  // the series: gaps over missing windows (more than 4 s apart)
  ctx.strokeStyle = props.color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  ctx.beginPath();
  let prev = null;
  ws.forEach((win, i) => {
    const v = vals[i];
    if (v === null) { prev = null; return; }
    const y = Y(Math.min(v, max));
    const joined = prev && win.t - prev.t <= 4;
    if (props.mode === 'step') {
      if (joined) { ctx.lineTo(X(win.t), Y(Math.min(prev.v, max))); ctx.lineTo(X(win.t), y); } else ctx.moveTo(X(win.t), y);
      ctx.lineTo(X(win.t + WINDOW_S), y);
    } else if (joined) ctx.lineTo(X(win.t), y);
    else ctx.moveTo(X(win.t), y);
    prev = { t: win.t, v };
  });
  ctx.stroke();
  if (props.mode === 'line') {
    ctx.fillStyle = props.color;
    ws.forEach((win, i) => {
      if (vals[i] === null) return;
      ctx.beginPath(); ctx.arc(X(win.t), Y(Math.min(vals[i], max)), 2.5, 0, 2 * Math.PI); ctx.fill();
    });
  }

  // hover: crosshair, highlighted point, tooltip
  const hw = hoverT === null ? null : nearest(hoverT);
  if (!hw) { tip.value.show = false; return; }
  const x = X(hw.t) + (props.mode === 'step' ? (X(hw.t + WINDOW_S) - X(hw.t)) / 2 : 0);
  ctx.strokeStyle = '#c4c7c5'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, h - PAD.b); ctx.stroke();
  const v = props.value(hw);
  const y = v === null ? null : Y(Math.min(v, max));
  if (y !== null) {
    ctx.fillStyle = props.color; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }
  if (props.tooltip) {
    tip.value = { show: true, html: props.tooltip(hw), left: Math.min(x + 14, w - 260), top: Math.max(0, (y ?? 40) - 24) };
  }
}

function onMove(e) {
  const r = canvas.value.getBoundingClientRect();
  hoverT = ((e.clientX - r.left - PAD.l) / (r.width - PAD.l - PAD.r)) * xMax();
  draw();
}
function onLeave() {
  hoverT = null;
  draw();
}

onMounted(() => {
  observer = new ResizeObserver(draw);
  observer.observe(canvas.value);
  watch([windows, tReplay, replayEvents], draw, { immediate: true });
});
onBeforeUnmount(() => observer?.disconnect());
</script>

<template>
  <div class="chart">
    <canvas ref="canvas" :style="{ height: `${height}px` }" @mousemove="onMove" @mouseleave="onLeave"></canvas>
    <!-- eslint-disable-next-line vue/no-v-html -- tooltip text is built from numbers and node names -->
    <div v-if="tip.show" class="tip" :style="{ left: `${tip.left}px`, top: `${tip.top}px` }" v-html="tip.html"></div>
  </div>
</template>

<style scoped>
.chart { position: relative; padding: 4px 8px 6px; }
canvas { display: block; width: 100%; }
.tip {
  position: absolute; pointer-events: none; z-index: 2; white-space: nowrap; background: #fff;
  border: 1.5px solid var(--rule); border-radius: 4px; padding: 6px 9px; font-size: 0.78rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
}
.tip :deep(b) { font-family: var(--font-mono); }
</style>
