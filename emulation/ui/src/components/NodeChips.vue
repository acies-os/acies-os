<script setup>
// Toggle chips for picking links: one per node, the edge uplink, and "All" for every node.
const props = defineProps({
  nodes: { type: Array, required: true }, // emulated node names
  modelValue: { type: Array, required: true }, // selected links
});
const emit = defineEmits(['update:modelValue']);

const allOn = () => props.nodes.length > 0 && props.nodes.every((n) => props.modelValue.includes(n));

function toggle(name) {
  const sel = props.modelValue;
  emit('update:modelValue', sel.includes(name) ? sel.filter((x) => x !== name) : [...sel, name]);
}
function toggleAll() {
  const sel = props.modelValue;
  emit('update:modelValue', allOn() ? sel.filter((x) => x === 'edge') : [...new Set([...props.nodes, ...sel])]);
}
</script>

<template>
  <span class="chips">
    <button type="button" class="all" :class="{ on: allOn() }" @click="toggleAll">All</button>
    <button
      v-for="name in [...nodes, 'edge']" :key="name" type="button"
      :class="{ on: modelValue.includes(name) }" @click="toggle(name)"
    >{{ name }}</button>
  </span>
</template>

<style scoped>
.chips { display: inline-flex; flex-wrap: wrap; gap: 3px; justify-content: center; }
button {
  padding: 0 6px; height: 22px; border: 1px solid var(--rule); border-radius: 3px; background: #fff;
  font: 600 0.66rem var(--font-mono); text-transform: uppercase; cursor: pointer;
}
button.on { background: #111; color: #fff; }
button.all { border-style: dashed; }
</style>
