<script setup>
// SPAR performance: summary over the last 30 s of replay, the error of every window, and how
// many nodes fed each window.
import { computed } from 'vue';
import { useMonitor } from '../composables/useMonitor.js';
import StatStrip from './StatStrip.vue';
import TimeChart from './TimeChart.vue';

const { snap, nodes } = useMonitor();

const stats = computed(() => {
  const s = snap.value?.spar || {};
  const unit = (v, u) => (v === null || v === undefined ? null : `${v} ${u}`);
  return [
    { label: 'Median error', value: unit(s.error_median_30s, 'm') },
    { label: 'P90 error', value: unit(s.error_p90_30s, 'm') },
    { label: 'Latest error', value: unit(s.last?.error_m, 'm') },
    { label: 'Nodes / window', value: s.nodes_mean_30s },
    { label: 'Windows', value: s.windows_30s === undefined ? null : `${s.windows_30s} / ${s.expected_30s}` },
    { label: 'Output lag', value: unit(s.lag_30s, 's') },
  ];
});

const errorTip = (w) =>
  `Window <b>${w.t}–${w.t + 2} s</b><br>Error <b>${w.error_m ?? '–'} m</b> vs truth at ${w.t} s<br>` +
  `${w.nodes.length} nodes: ${w.nodes.join(' ').toUpperCase()}<br>Lag <b>${w.lag_s ?? '–'} s</b> · ${w.label.toUpperCase()} ${w.score}`;
</script>

<template>
  <section class="panel">
    <StatStrip :items="stats" big />
    <div class="panel-head">
      <span class="panel-title">SPAR localization error</span>
      <span class="panel-note">Distance from the GPS truth at the start of each 2-s window, the position SPAR predicts. Stats: last 30 s of replay.</span>
      <span class="legend"><span><i></i>Error (m)</span><span><i class="dash"></i>Link change</span></span>
    </div>
    <TimeChart :value="(w) => w.error_m" :y-format="(v) => `${v} m`" :tooltip="errorTip" :height="230" />
    <div class="panel-head divider">
      <span class="panel-title">Nodes in each window</span>
      <span class="panel-note">Nodes whose data reached the window before SPAR closed it.</span>
    </div>
    <TimeChart
      :value="(w) => w.nodes.length" mode="step" color="#111" :height="110"
      :y-max="Math.max(1, nodes.length)" :labels="false"
    />
  </section>
</template>

<style scoped>
.divider { border-top: 1px solid #ddd; }
.legend { display: flex; gap: 14px; font-size: 0.78rem; font-weight: 600; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { display: inline-block; width: 16px; border-top: 2px solid var(--accent); }
.legend i.dash { border-top: 2px dashed #111; }
</style>
