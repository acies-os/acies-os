<script setup>
// A row of labelled values, as the main UI's STATUS / MAP / NODES / TARGET strip.
// items: [{label, value, tone?: 'ok'}]; `big` makes the values larger.
defineProps({
  items: { type: Array, required: true },
  big: { type: Boolean, default: false },
});
const show = (v) => (v === null || v === undefined || v === '' ? '–' : v);
</script>

<template>
  <div class="stats">
    <div v-for="item in items" :key="item.label" class="stat">
      <span class="label">{{ item.label }}</span>
      <span class="value" :class="{ ok: item.tone === 'ok', big }">{{ show(item.value) }}</span>
    </div>
  </div>
</template>

<style scoped>
.stats { display: flex; border-bottom: 1.5px solid var(--rule); }
.stat {
  flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  padding: 8px 12px; gap: 2px; border-right: 1px solid #e0e0e0; min-width: 0;
}
.stat:last-child { border-right: none; }
.label { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: #000; white-space: nowrap; }
.value { font: 500 1rem var(--font-mono); color: var(--accent); text-align: center; white-space: nowrap; }
.value.ok { color: var(--good); }
.value.big { font-size: 1.35rem; }
@media (max-width: 900px) {
  .stats { flex-wrap: wrap; }
}
</style>
