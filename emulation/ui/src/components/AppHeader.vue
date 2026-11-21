<script setup>
// Header as on the main UI's pages, with a link back to the main UI (port 5173).
const mainUi = `${location.protocol}//${location.hostname}:5173/`;
</script>

<template>
  <header class="header">
    <div class="left">
      <img src="/uiuclogo.png" class="logo" alt="UIUC" />
      <div class="title">ACIES-OS</div>
      <div class="sub">Network Emulator</div>
    </div>
    <div class="right">
      <a class="pill" :href="mainUi">Main UI</a>
    </div>
  </header>
</template>

<style scoped>
.header {
  margin: 0 2.5%; height: 100px; width: 95%; display: flex; align-items: center;
  justify-content: space-between; flex-wrap: wrap; gap: 8px; border-bottom: 2px solid var(--rule);
}
.left { display: flex; align-items: center; gap: 14px; padding-left: 2.5%; }
.logo { height: 2.5rem; width: auto; }
.title {
  font-family: var(--font-title); font-size: max(min(2rem, 2vw), 1.5rem); font-weight: 500;
  letter-spacing: 0.2em; color: #222; text-transform: uppercase;
}
.sub {
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--accent); border-left: 1.5px solid var(--rule); padding-left: 14px;
}
.right { padding-right: 2.5%; }
@media (max-width: 900px) {
  .header { height: auto; padding: 12px 0; }
}
</style>
