<script setup>
// Every link change the emulator applied, newest first.
import { computed } from 'vue';
import { useMonitor } from '../composables/useMonitor.js';

const { snap } = useMonitor();

const entries = computed(() => {
  const s = snap.value;
  return (s?.events || [])
    .filter((e) => e.action !== 'plan')
    .slice()
    .reverse()
    .map((e) => ({
      key: `${e.wall_ns}-${e.action}`,
      // seconds after the current replay's start, else the clock time
      time: s.start_at && e.start_at === s.start_at
        ? `${Math.round(e.wall_ns / 1e9 - s.start_at)} s`
        : new Date(e.wall_ns / 1e6).toLocaleTimeString(),
      text: [e.action, (e.nodes || []).join(', ').toUpperCase(),
        Object.entries(e.params || {}).map(([k, v]) => `${k}=${v}`).join(' ')].filter(Boolean).join(' '),
      source: e.source,
    }));
});
</script>

<template>
  <section class="panel">
    <div class="panel-head">
      <span class="panel-title">Applied changes</span>
      <span class="panel-note">Every change is recorded for <code>just emu-analyze</code></span>
    </div>
    <ul>
      <li v-for="e in entries" :key="e.key">
        <span class="time">{{ e.time }}</span>{{ e.text }} <span class="source">({{ e.source }})</span>
      </li>
      <li v-if="!entries.length" class="none">None yet</li>
    </ul>
  </section>
</template>

<style scoped>
ul { list-style: none; margin: 0; padding: 0; max-height: 220px; overflow: auto; font-size: 0.82rem; scrollbar-width: thin; }
li { padding: 5px 12px; border-bottom: 1px solid var(--rule-soft); }
.time { font-family: var(--font-mono); color: var(--accent); display: inline-block; min-width: 60px; margin-right: 14px; }
.source { color: var(--muted); }
.none { color: #999; }
</style>
