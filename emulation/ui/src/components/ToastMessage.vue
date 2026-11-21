<script setup>
// The monitor's reply to the last command ("sent down to ['rs3']", or an error), for 6 s.
import { ref, watch } from 'vue';
import { useMonitor } from '../composables/useMonitor.js';

const { state } = useMonitor();
const shown = ref(null);
let timer = null;

watch(() => state.toast, (t) => {
  if (!t.text) return;
  shown.value = t;
  clearTimeout(timer);
  timer = setTimeout(() => { shown.value = null; }, 6000);
});
</script>

<template>
  <div v-if="shown" class="toast" :class="{ error: shown.error }">{{ shown.text }}</div>
</template>

<style scoped>
.toast {
  position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%); z-index: 1500;
  background: #111; color: #fff; padding: 8px 16px; border-radius: 4px;
  font: 500 0.8rem var(--font-mono); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.toast.error { background: var(--accent); }
</style>
