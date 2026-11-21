<script setup>
// The planned network changes: one row per change (when, which links, what, value, how long).
import { computed } from 'vue';
import { useExperiment } from '../composables/useExperiment.js';
import { useMonitor } from '../composables/useMonitor.js';
import { CHANGES, newRow, presets } from '../lib/plan.js';
import NodeChips from './NodeChips.vue';

const { plan, applyPlan } = useExperiment();
const { nodes, tReplay } = useMonitor();

const running = computed(() => tReplay.value !== null && tReplay.value >= 0);
const isDone = (row) => running.value && tReplay.value >= Number(row.at) + (row.dur ? Number(row.dur) : 0);

function add() {
  const last = plan.value[plan.value.length - 1];
  plan.value.push(newRow(last ? Number(last.at) + 30 : 30));
}
function setChange(row, change) {
  row.change = change;
  if (!CHANGES[change].unit) row.value = '';
}
function addPreset(name) {
  plan.value.push(...presets(nodes.value)[name]);
}
</script>

<template>
  <div class="head">
    <span class="sub-title">Planned network changes</span>
    <span class="panel-note">Seconds after the replay starts; they fire automatically. Leave "For" empty to keep a change until the end.</span>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th>At (s)</th><th>Links</th><th>Change</th><th>Value</th><th>For (s)</th><th></th></tr>
      </thead>
      <tbody>
        <tr v-for="(row, i) in plan" :key="i" :class="{ done: isDone(row) }">
          <td><input v-model.number="row.at" class="field" type="number" min="0" step="1" /></td>
          <td><NodeChips v-model="row.nodes" :nodes="nodes" /></td>
          <td>
            <select class="field" :value="row.change" @change="setChange(row, $event.target.value)">
              <option v-for="(kind, key) in CHANGES" :key="key" :value="key">{{ kind.label }}</option>
            </select>
          </td>
          <td>
            <input
              v-model.number="row.value" class="field" type="number" min="0" step="any"
              :placeholder="CHANGES[row.change].unit || '–'" :disabled="!CHANGES[row.change].unit"
            />
          </td>
          <td>
            <input
              v-model.number="row.dur" class="field" type="number" min="0" step="1"
              :placeholder="CHANGES[row.change].revert ? 'to end' : '–'" :disabled="!CHANGES[row.change].revert"
            />
          </td>
          <td><button class="remove" type="button" title="Remove" @click="plan.splice(i, 1)">×</button></td>
        </tr>
      </tbody>
    </table>
    <div v-if="!plan.length" class="empty">No planned changes: the replay runs on healthy links.</div>
  </div>
  <div class="actions">
    <button class="btn btn--dark" type="button" @click="add">+ Add change</button>
    <span class="preset-label">Presets</span>
    <button class="btn" type="button" @click="addPreset('blackout')">3 links black out</button>
    <button class="btn" type="button" @click="addPreset('loss')">20% loss on all</button>
    <button class="btn" type="button" @click="addPreset('edge')">Edge uplink 1 Mbit/s</button>
    <button class="btn" type="button" @click="plan.splice(0)">Clear</button>
    <button
      v-if="running" class="btn btn--accent apply" type="button"
      title="Replace the running replay's schedule with this plan" @click="applyPlan"
    >Apply to running replay</button>
  </div>
</template>

<style scoped>
.head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 12px; padding: 10px 12px 6px; }
td { border-bottom: 1px solid var(--rule-soft); }
tr.done td { color: #999; background: #fafafa; }
input.field { width: 70px; }
.remove { border: none; background: none; font-size: 1.1rem; line-height: 1; cursor: pointer; color: #999; }
.remove:hover { color: var(--accent); }
.empty { padding: 10px 12px; color: var(--muted); font-size: 0.85rem; }
.actions {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; padding: 10px 12px;
  border-top: 1px solid #ddd;
}
.preset-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-left: 8px; }
.apply { margin-left: auto; }
</style>
