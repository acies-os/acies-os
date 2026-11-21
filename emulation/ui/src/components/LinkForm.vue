<script setup>
// Set bandwidth / delay / jitter / loss on one link, all nodes or the edge uplink, now.
import { reactive, ref } from 'vue';
import { useMonitor } from '../composables/useMonitor.js';

const { nodes, fault, toast } = useMonitor();
const target = ref('all');
const values = reactive({ bw: '', delay: '', jitter: '', loss: '' });

const targetNodes = () => (target.value === 'all' ? 'all' : [target.value]);

function apply() {
  const params = {};
  if (values.bw !== '') params.bw = Number(values.bw);
  if (values.delay !== '') params.delay = `${Number(values.delay)}ms`;
  if (values.jitter !== '') params.jitter = `${Number(values.jitter)}ms`;
  if (values.loss !== '') params.loss = Number(values.loss);
  if (!Object.keys(params).length) {
    toast('enter at least one value', true);
    return;
  }
  fault('set', targetNodes(), params);
}
</script>

<template>
  <form class="form" @submit.prevent="apply">
    <label class="field-label">Link
      <select v-model="target" class="field">
        <option value="all">All nodes</option>
        <option v-for="n in [...nodes, 'edge']" :key="n" :value="n">{{ n.toUpperCase() }}</option>
      </select>
    </label>
    <label class="field-label">BW <input v-model="values.bw" class="field" type="number" min="0" step="any" placeholder="Mb/s" /></label>
    <label class="field-label">Delay <input v-model="values.delay" class="field" type="number" min="0" step="any" placeholder="ms" /></label>
    <label class="field-label">Jitter <input v-model="values.jitter" class="field" type="number" min="0" step="any" placeholder="ms" /></label>
    <label class="field-label">Loss <input v-model="values.loss" class="field" type="number" min="0" max="100" step="any" placeholder="%" /></label>
    <button class="btn btn--dark" type="submit">Apply</button>
    <button class="btn" type="button" @click="fault('reset', targetNodes())">Reset</button>
    <button class="btn btn--accent" type="button" @click="fault('reset', 'all+edge')">Reset all</button>
  </form>
</template>

<style scoped>
.form { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; padding: 10px 12px; border-top: 1.5px solid var(--rule); }
input.field { width: 62px; }
</style>
