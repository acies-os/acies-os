<script setup>
// Satellite map (Esri imagery, as in the main UI): nodes coloured by link state (click one to
// black it out or restore it), the vehicle's GPS truth (drawn 2 s late by the monitor, like the
// main UI) and SPAR's last predictions while SPAR is running.
import L from 'leaflet';
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { TILE_URL, targetColor } from '../config.js';
import { useMonitor } from '../composables/useMonitor.js';
import { linkStatus } from '../lib/links.js';

const { state, snap, links, windows, sparRunning, fault } = useMonitor();
const el = ref(null);
let map = null;
let layers = null;
let nodeMarkers = {};

// the site the gateway is using
const site = computed(() => state.sites[snap.value?.map] || null);
// vehicle label for the legend: the truth's, else SPAR's
const vehicle = computed(() => Object.keys(snap.value?.truth_trail || {})[0] || windows.value[windows.value.length - 1]?.label || '');

function nodeIcon(name, status, west) {
  const tag = status.cls !== 'up' && status.cls !== 'unknown' ? `<small>${status.label}</small>` : '';
  const shift = west ? ' style="transform: translateX(-100%)"' : ''; // west column: label to the left
  return L.divIcon({
    className: '', iconSize: null, iconAnchor: [0, 10],
    html: `<div class="node-label st-${status.cls}"${shift}>${name}${tag}</div>`,
  });
}

function drawSite() {
  layers.road.clearLayers();
  layers.nodes.clearLayers();
  nodeMarkers = {};
  const s = site.value;
  if (!s) return;
  const points = [...Object.values(s.nodes), ...s.road];
  if (points.length) map.fitBounds(L.latLngBounds(points), { padding: [40, 40] });
  if (s.road.length) L.polyline(s.road, { color: '#fff', weight: 2, opacity: 0.5, dashArray: '6 6' }).addTo(layers.road);
  const lons = Object.values(s.nodes).map((p) => p[1]);
  const midLon = lons.reduce((a, b) => a + b, 0) / Math.max(1, lons.length);
  for (const [name, pos] of Object.entries(s.nodes)) {
    const west = pos[1] < midLon;
    const marker = L.marker(pos, { icon: nodeIcon(name, linkStatus(null), west) })
      .on('click', () => {
        const link = links.value[name];
        if (link) fault(link.down ? 'up' : 'down', [name]);
      })
      .bindTooltip('Click: blackout on / off', { direction: 'top' })
      .addTo(layers.nodes);
    nodeMarkers[name] = { marker, west, label: null };
  }
}

function drawState() {
  for (const [name, m] of Object.entries(nodeMarkers)) {
    const status = linkStatus(links.value[name]);
    if (m.label !== status.label) { // only on change: no flicker, tooltip stays
      m.label = status.label;
      m.marker.setIcon(nodeIcon(name, status, m.west));
    }
  }
  layers.truth.clearLayers();
  for (const [label, trail] of Object.entries(snap.value?.truth_trail || {})) {
    if (!trail.length) continue;
    const c = targetColor(label);
    L.polyline(trail, { color: '#000', weight: 5, opacity: 0.6 }).addTo(layers.truth);
    L.polyline(trail, { color: c, weight: 3 }).addTo(layers.truth);
    L.circleMarker(trail[trail.length - 1], { radius: 7, color: '#000', weight: 2, fillColor: c, fillOpacity: 1 })
      .bindTooltip(`${label.toUpperCase()} truth (shown 2 s late, like the main UI)`)
      .addTo(layers.truth);
  }
  layers.spar.clearLayers();
  const recent = sparRunning.value ? windows.value.slice(-8) : [];
  recent.forEach((w, i) => {
    const latest = i === recent.length - 1;
    const size = latest ? 16 : 10;
    L.marker([w.lat, w.lon], {
      opacity: 0.35 + (0.65 * (i + 1)) / recent.length,
      icon: L.divIcon({ className: '', iconSize: [size, size], html: `<div class="spar-mark" style="--c:${targetColor(w.label)}"></div>` }),
    })
      .bindTooltip(`SPAR ${w.t}–${w.t + 2} s · error ${w.error_m ?? '–'} m · ${w.nodes.length} nodes`)
      .addTo(layers.spar);
  });
}

onMounted(() => {
  map = L.map(el.value, { maxZoom: 22, attributionControl: false }).setView([40.289, -88.1263], 18);
  L.tileLayer(TILE_URL, { maxZoom: 22, maxNativeZoom: 18 }).addTo(map);
  layers = {
    road: L.layerGroup().addTo(map),
    truth: L.layerGroup().addTo(map),
    spar: L.layerGroup().addTo(map),
    nodes: L.layerGroup().addTo(map),
  };
  watch(site, drawSite, { immediate: true });
  watch(snap, drawState, { immediate: true });
});
onBeforeUnmount(() => map?.remove());
</script>

<template>
  <div class="wrapper">
    <div ref="el" class="map"></div>
    <div class="legend">
      <span><i class="gt" :style="{ '--c': targetColor(vehicle) }"></i>{{ vehicle.toUpperCase() }} truth</span>
      <span v-if="sparRunning"><i class="spar" :style="{ '--c': targetColor(vehicle) }"></i>{{ vehicle.toUpperCase() }} SPAR</span>
      <span><i class="link" style="--c: var(--map-up)"></i>Link up</span>
      <span><i class="link" style="--c: var(--map-impaired)"></i>Impaired</span>
      <span><i class="link" style="--c: var(--map-down)"></i>Down / unplugged</span>
    </div>
  </div>
</template>

<style scoped>
.wrapper { position: relative; border-radius: 10px; overflow: hidden; }
.map { width: 100%; height: 100%; background: #ddd; }
.legend {
  position: absolute; top: 10px; right: 10px; z-index: 500; background: rgba(255, 255, 255, 0.92);
  border: 1.5px solid var(--rule); border-radius: 4px; padding: 6px 10px; font-size: 0.78rem; font-weight: 600;
  display: flex; flex-direction: column; gap: 2px;
}
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 10px; height: 10px; background: var(--c); }
.legend .gt { border-radius: 50%; border: 2px solid #000; }
.legend .spar { border: 2px dashed #000; }
.legend .link { border-radius: 2px; border: 1px solid #000; }

/* markers are created by Leaflet outside this component's DOM, hence :deep */
:deep(.node-label) {
  display: inline-flex; align-items: baseline; gap: 4px; border: 1.5px solid #000; border-radius: 3px;
  padding: 1px 6px; font: 700 0.78rem var(--font-text); letter-spacing: 0.04em; text-transform: uppercase;
  white-space: nowrap; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.5); cursor: pointer; color: #fff;
}
:deep(.node-label small) { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em; }
:deep(.node-label.st-unknown) { background: var(--map-unknown); }
:deep(.node-label.st-up) { background: var(--map-up); }
:deep(.node-label.st-impaired) { background: var(--map-impaired); color: #111; }
:deep(.node-label.st-down) { background: var(--map-down); }
:deep(.spar-mark) { width: 100%; height: 100%; border: 2px dashed #000; background: var(--c); }
</style>
