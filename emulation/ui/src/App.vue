<script setup>
// Page layout. State lives in the composables (useMonitor, useExperiment, useDeployOverlay);
// each panel below reads what it needs from them.
import AppHeader from './components/AppHeader.vue';
import ChangeLog from './components/ChangeLog.vue';
import DeployOverlay from './components/DeployOverlay.vue';
import ExperimentPanel from './components/ExperimentPanel.vue';
import LinkPanel from './components/LinkPanel.vue';
import NetworkMap from './components/NetworkMap.vue';
import SparPanel from './components/SparPanel.vue';
import ToastMessage from './components/ToastMessage.vue';
</script>

<template>
  <AppHeader />
  <main class="content">
    <ExperimentPanel />
    <div class="row">
      <NetworkMap class="map" />
      <LinkPanel class="links" />
    </div>
    <SparPanel />
    <ChangeLog />
  </main>
  <DeployOverlay />
  <ToastMessage />
  <div class="copyright">© {{ new Date().getFullYear() }} UIUC CyPhy Group. All rights reserved.</div>
</template>

<style scoped>
.content { width: 95%; margin: 24px auto 48px; display: flex; flex-direction: column; gap: 24px; }
.row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
.map { flex: 1 1 420px; min-width: 0; height: 560px; }
.links { flex: 0 1 clamp(680px, 52vw, 860px); min-width: 0; }
.copyright {
  position: fixed; bottom: 10px; right: 14px; font-size: 0.68rem; font-weight: 500;
  color: #999; pointer-events: none;
}
@media (max-width: 900px) {
  .links { flex-basis: 100%; }
}
</style>
