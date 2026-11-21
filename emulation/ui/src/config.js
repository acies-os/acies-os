// Settings that are UI choices rather than data from the monitor.

// Vehicle colours, from ui/app/src/config/app.js (TARGET_COLORS).
export const TARGET_COLORS = {
  gle350: '#E05339', suv: '#E05339', miata: '#816EE0', mustang: '#06b6d4', cx30: '#f59e0b',
  polaris: '#ec4899', warthog: '#8b5cf6', mazdacx30: '#84cc16', truck: '#fb7185', wagoneer: '#7079F3',
  kona: '#36A553', others: '#36A553',
};
export const targetColor = (label) => TARGET_COLORS[String(label || '').toLowerCase()] || '#E05339';

// Targets offered per site, as in the main UI (ui/app/src/config/maps.js MAP_TARGETS).
// A site not listed here offers all its routes.
export const SITE_TARGETS = { ict: ['gle350', 'cx30'], wp: ['wagoneer', 'kona'] };
export const TARGET_LABELS = { kona: 'Others' };
export const DEFAULT_TARGET = 'gle350';

// SPAR counts as running while its newest window is this close to the replay time.
export const SPAR_STALE_S = 10;

// Esri imagery, as in the main UI.
export const TILE_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
