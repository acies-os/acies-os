<script setup>
// The configuring window, styled as the main UI's "Deploying configuration…" overlay.
import { useDeployOverlay } from '../composables/useDeployOverlay.js';

const { visible, progress, message, dismiss } = useDeployOverlay();
</script>

<template>
  <div v-if="visible" class="overlay">
    <div class="box">
      <div class="spinner"></div>
      <div class="text">Deploying configuration…</div>
      <div class="bar"><div class="fill" :style="{ width: `${progress}%` }"></div></div>
      <div class="sub">{{ message }}</div>
      <button class="btn" type="button" @click="dismiss">Hide</button>
    </div>
  </div>
</template>

<style scoped>
.overlay {
  position: fixed; inset: 0; z-index: 2000; background: rgba(255, 255, 255, 0.95);
  display: flex; align-items: center; justify-content: center;
}
.box { display: flex; flex-direction: column; align-items: center; }
.spinner {
  width: 48px; height: 48px; border: 4px solid var(--accent-tonal); border-top-color: var(--accent);
  border-radius: 50%; animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.text { margin-top: 16px; font-size: 1rem; font-weight: 600; }
.bar { width: 260px; height: 4px; background: var(--accent-tonal); border-radius: 2px; overflow: hidden; margin-top: 16px; }
.fill { height: 100%; background: var(--accent); transition: width 0.3s linear; }
.sub { margin: 8px 0 18px; font-size: 0.82rem; color: #555; text-align: center; }
</style>
