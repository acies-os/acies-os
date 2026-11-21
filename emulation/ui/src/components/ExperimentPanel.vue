<script setup>
// Experiment setup: site and target, the plan, and Deploy.
import { computed } from 'vue';
import { useDeployOverlay } from '../composables/useDeployOverlay.js';
import { useExperiment } from '../composables/useExperiment.js';
import { useMonitor } from '../composables/useMonitor.js';
import PlanTable from './PlanTable.vue';

const exp = useExperiment();
const overlay = useDeployOverlay();
const { tReplay, windows } = useMonitor();

const status = computed(() => {
  const t = tReplay.value;
  if (t === null) return 'Not deployed';
  if (t < 0) return `Starting in ${Math.ceil(-t)} s`;
  const label = windows.value[windows.value.length - 1]?.label?.toUpperCase() || '';
  return `Replaying ${label} · ${Math.round(t)} s`;
});

function deploy() {
  if (exp.deploy()) overlay.begin();
}
</script>

<template>
  <section class="panel">
    <div class="panel-head">
      <span class="panel-title">Experiment</span>
      <span class="panel-note">1. Pick the site and target. 2. Plan the network changes (optional). 3. Deploy. You can also change links live below.</span>
    </div>
    <div class="setup">
      <label>Site
        <select :value="exp.site.value" @change="exp.pick($event.target.value)">
          <option v-for="s in exp.sites.value" :key="s" :value="s">{{ s.toUpperCase() }}</option>
        </select>
      </label>
      <label>Target
        <select :value="exp.target.value" @change="exp.pick(undefined, $event.target.value)">
          <option v-for="t in exp.targets.value" :key="t.value" :value="t.value" :disabled="t.disabled">{{ t.label }}</option>
        </select>
      </label>
      <span class="status">{{ status }}</span>
      <button class="pill pill--filled" type="button" @click="deploy">Deploy SPAR</button>
    </div>
    <PlanTable />
  </section>
</template>

<style scoped>
.setup { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; padding: 12px; border-bottom: 1px solid #ddd; }
label { display: inline-flex; align-items: center; gap: 8px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; }
select {
  height: 36px; min-width: 150px; padding: 0 10px; border: 1.5px solid var(--rule); border-radius: 4px;
  font: 500 0.9rem var(--font-mono); color: var(--ink); background: #fff; cursor: pointer;
}
.status { font: 500 0.95rem var(--font-mono); color: var(--accent); margin-left: auto; }
</style>
