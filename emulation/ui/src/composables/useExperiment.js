// Experiment setup: site, target and the fault plan, and deploying them.
// Shared by the Experiment panel and the map (one store for the app).
import { computed, ref, watch } from 'vue';
import { DEFAULT_TARGET, SITE_TARGETS, TARGET_LABELS } from '../config.js';
import { eventsFromRows, rowsFromEvents } from '../lib/plan.js';
import { load, save } from '../lib/storage.js';
import { useMonitor } from './useMonitor.js';

const site = ref('');
const target = ref(load('target', ''));
const touched = ref(false); // the user picked a site/target: stop following the running one
const plan = ref([]);
let initialised = false;

export function useExperiment() {
  const mon = useMonitor();

  // Only sites whose node list covers the emulated nodes (e.g. ICT and West Point, not GQ).
  const sites = computed(() => {
    const nodes = mon.nodes.value;
    return Object.keys(mon.state.sites).filter((k) => nodes.length && nodes.every((n) => n in mon.state.sites[k].nodes));
  });

  // The site's targets as in the main UI; ones without recordings on this server are disabled.
  const targets = computed(() => {
    const routes = mon.state.sites[site.value]?.targets || {};
    const need = mon.nodes.value.length;
    return (SITE_TARGETS[site.value] || Object.keys(routes))
      .filter((k) => k in routes)
      .map((k) => {
        const have = routes[k].nodes_with_data;
        const note = have === 0 ? ' (no data)' : have < need ? ` (${have}/${need} nodes)` : '';
        return { value: k, label: (TARGET_LABELS[k] || k).toUpperCase().replace(/_/g, ' + ') + note, disabled: have === 0 };
      });
  });

  // keep the site valid; follow the gateway's site until the user picks one
  watch([sites, () => mon.snap.value?.map], ([list, active]) => {
    if (!list.length) return;
    if (!touched.value && list.includes(active)) site.value = active;
    else if (!list.includes(site.value)) site.value = list[0];
  }, { immediate: true });

  // keep the target valid: the user's choice, else GLE350, else the first with data
  watch(targets, (opts) => {
    const usable = opts.filter((o) => !o.disabled).map((o) => o.value);
    if (usable.includes(target.value)) return;
    target.value = usable.includes(DEFAULT_TARGET) ? DEFAULT_TARGET : usable[0] || '';
  }, { immediate: true });

  // show the target that is playing: the route whose vehicles match the GPS truth's labels
  watch(() => mon.snap.value?.truth_trail, (trail) => {
    if (touched.value || !trail || site.value !== mon.snap.value?.map) return;
    const labels = Object.keys(trail).map((l) => l.toLowerCase()).sort().join(',');
    const match = targets.value.find((o) => !o.disabled && o.value.split('_').sort().join(',') === labels);
    if (match) target.value = match.value;
  });

  // the plan starts from this browser's last plan, else the emulator's current schedule
  watch(() => mon.snap.value, (s) => {
    if (initialised || !s) return;
    initialised = true;
    const saved = load('plan');
    plan.value = Array.isArray(saved) ? saved : rowsFromEvents(s.schedule || [], s.nodes || []);
  }, { immediate: true });
  watch(plan, (rows) => save('plan', rows), { deep: true });

  function pick(newSite, newTarget) {
    touched.value = true;
    if (newSite !== undefined) site.value = newSite;
    if (newTarget !== undefined) {
      target.value = newTarget;
      save('target', newTarget);
    }
  }

  function deploy() {
    if (!target.value) {
      mon.toast('pick a target with data', true);
      return false;
    }
    const ok = mon.send({ type: 'deploy', target: target.value, map: site.value, plan: eventsFromRows(plan.value) });
    if (ok) touched.value = false;
    return ok;
  }

  const applyPlan = () => mon.send({ type: 'plan', plan: eventsFromRows(plan.value) });

  return { site, target, plan, sites, targets, pick, deploy, applyPlan };
}
