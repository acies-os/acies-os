// Small canvas helpers for the time charts (no chart library needed for two line charts).

export const PAD = { l: 46, r: 14, t: 22, b: 22 };

/** Size a canvas for the device pixel ratio; returns [ctx, width, height] in CSS pixels. */
export function setupCanvas(canvas, height) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = '500 11px "IBM Plex Mono", monospace';
  return [ctx, w, height];
}

/** 1, 2, 2.5, 5 x 10^n at or above v. */
export function niceStep(v) {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

/** Gridlines, y labels, x labels (seconds) and the baseline; returns the scales X(t), Y(v). */
export function drawAxes(ctx, w, h, xMax, yMax, yTicks, yFormat) {
  const X = (t) => PAD.l + (t / xMax) * (w - PAD.l - PAD.r);
  const Y = (v) => h - PAD.b - (v / yMax) * (h - PAD.t - PAD.b);
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#ececec';
  ctx.fillStyle = '#888';
  ctx.textAlign = 'right';
  for (const v of yTicks) {
    ctx.beginPath(); ctx.moveTo(PAD.l, Y(v)); ctx.lineTo(w - PAD.r, Y(v)); ctx.stroke();
    ctx.fillText(yFormat(v), PAD.l - 6, Y(v) + 4);
  }
  ctx.textAlign = 'center';
  const step = niceStep(xMax / 8);
  for (let t = 0; t <= xMax; t += step) ctx.fillText(`${t}s`, X(t), h - 5);
  ctx.textAlign = 'left';
  ctx.strokeStyle = '#111';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(PAD.l, Y(0)); ctx.lineTo(w - PAD.r, Y(0)); ctx.stroke();
  return [X, Y];
}

/** Dashed line per link change (labelled when `labels`), and a solid line at "now". */
export function drawMarks(ctx, X, h, events, tNow, labels) {
  ctx.font = '700 10px Barlow, sans-serif';
  let row = 0;
  let lastX = -1e9;
  for (const e of events) {
    const x = X(e.t);
    ctx.strokeStyle = '#111'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x, PAD.t - 8); ctx.lineTo(x, h - PAD.b); ctx.stroke();
    ctx.setLineDash([]);
    if (labels) {
      row = x - lastX < 150 ? (row + 1) % 2 : 0; // stagger labels that would overlap
      lastX = x;
      const nodes = (e.nodes || []).length > 3 ? `${e.nodes.length} LINKS` : (e.nodes || []).join(' ').toUpperCase();
      ctx.fillStyle = '#111';
      ctx.fillText(`${e.action.toUpperCase()} ${nodes}`, x + 4, 9 + row * 11);
    }
  }
  ctx.font = '500 11px "IBM Plex Mono", monospace';
  if (tNow > 0) {
    ctx.strokeStyle = '#c4c7c5'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(tNow), PAD.t - 8); ctx.lineTo(X(tNow), h - PAD.b); ctx.stroke();
  }
}
