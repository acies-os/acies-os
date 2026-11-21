<script setup>
// One row per link: state, settings, what was measured over the last 10 s, and live fault buttons.
import { computed } from 'vue';
import { useMonitor } from '../composables/useMonitor.js';
import { delayText, linkStatus } from '../lib/links.js';

const { nodes, links, fault } = useMonitor();

const rows = computed(() => [...nodes.value, 'edge'].map((name) => {
  const link = links.value[name] || {};
  return { name, link, edge: name === 'edge', status: linkStatus(link), config: link.config || {} };
}));
const show = (v, suffix = '') => (v === null || v === undefined || v === '' ? null : `${v}${suffix}`);
</script>

<template>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Node</th>
          <th rowspan="2">State</th>
          <th class="group" colspan="3">Link settings</th>
          <th class="group" colspan="5">Measured (last 10 s)</th>
          <th rowspan="2">Fault</th>
        </tr>
        <tr>
          <th>BW Mb/s</th><th>Delay</th><th>Loss %</th>
          <th>Up kb/s</th><th>Drops/s</th><th>Delivered</th><th>Extra delay</th><th>Last data</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in rows" :key="r.name" :class="{ edge: r.edge }">
          <td class="name">{{ r.name }}</td>
          <td><span class="chip" :class="r.status.cls">{{ r.status.label }}</span></td>
          <td class="num">{{ show(r.config.bw) ?? '–' }}</td>
          <td class="num">{{ delayText(r.config) ?? '–' }}</td>
          <td class="num">{{ show(r.config.loss) ?? '–' }}</td>
          <td class="num">{{ show(r.link.up_kbps) ?? '–' }}</td>
          <td class="num">{{ show(r.link.up_drops_ps) ?? '–' }}</td>
          <td class="num">
            <template v-if="!r.edge && r.link.delivery_pct !== undefined">
              {{ r.link.delivery_pct }}%<span class="bar"><i :style="{ width: `${r.link.delivery_pct}%` }"></i></span>
            </template>
            <span v-else class="dim">–</span>
          </td>
          <td class="num">{{ r.edge ? '–' : show(r.link.delay_ms, ' ms') ?? '–' }}</td>
          <td class="num">{{ r.edge ? '–' : show(r.link.last_data_s, ' s') ?? '–' }}</td>
          <td>
            <span class="seg">
              <button
                type="button" :class="{ on: r.link.down }" title="100% loss; the interface stays up"
                @click="fault(r.link.down ? 'up' : 'down', [r.name])"
              >Down</button>
              <button
                type="button" :class="{ on: r.link.unplugged }" title="Interface down"
                @click="fault(r.link.unplugged ? 'plug' : 'unplug', [r.name])"
              >Unplug</button>
            </span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
th.group { padding: 4px 5px; border-bottom: 1px solid var(--rule-soft); }
td { border-bottom: 1px solid var(--rule-row); }
tbody tr:last-child td { border-bottom: none; }
tr.edge td { border-top: 1px dashed var(--rule); background: #fafafa; }
.name { font-weight: 700; font-size: 0.88rem; color: #555; text-transform: uppercase; }
.chip { display: inline-block; padding: 1px 8px; border-radius: 4px; border: 1px solid #000; font-weight: 600; font-size: 0.78rem; }
.chip.up { background: var(--good-bg); }
.chip.impaired { background: var(--warn-bg); }
.chip.down { background: var(--bad-bg); color: #7d2a13; }
.bar {
  display: inline-block; width: 38px; height: 6px; border: 1px solid var(--rule); border-radius: 3px;
  margin-left: 5px; vertical-align: middle; overflow: hidden;
}
.bar i { display: block; height: 100%; background: var(--rule); }
/* segmented buttons, as the main UI's computation-graph node switch */
.seg { display: inline-flex; border: 1.5px solid var(--rule); border-radius: 4px; overflow: hidden; vertical-align: middle; }
.seg button {
  margin: 0; padding: 1px 8px; border: none; border-right: 1px solid var(--rule); border-radius: 0;
  background: #fff; color: #111; font: 600 0.68rem var(--font-mono); letter-spacing: 0.05em;
  text-transform: uppercase; cursor: pointer;
}
.seg button:last-child { border-right: none; }
.seg button:hover { background: #f1f1f1; }
.seg button.on { background: #111; color: #fff; }
</style>
