<script setup>
// Status strip + link table + set form.
import { computed } from 'vue';
import { useMonitor } from '../composables/useMonitor.js';
import LinkForm from './LinkForm.vue';
import LinkTable from './LinkTable.vue';
import StatStrip from './StatStrip.vue';

const { state, snap, nodes, links, tReplay } = useMonitor();

const stats = computed(() => {
  const t = tReplay.value;
  const down = nodes.value.filter((n) => links.value[n]?.down || links.value[n]?.unplugged);
  return [
    { label: 'Status', value: state.connected ? 'Connected' : 'Disconnected', tone: state.connected ? 'ok' : '' },
    { label: 'Experiment', value: snap.value?.experiment },
    { label: 'Replay', value: t === null ? 'Not deployed' : t < 0 ? `Starts in ${Math.ceil(-t)} s` : `${Math.round(t)} s` },
    { label: 'Links down', value: down.length ? down.map((n) => n.toUpperCase()).join(' ') : 'None' },
  ];
});
</script>

<template>
  <section class="panel">
    <StatStrip :items="stats" />
    <LinkTable />
    <LinkForm />
  </section>
</template>
