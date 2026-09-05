const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (v, d = 1) => v == null ? "–" : (v * 100).toFixed(d) + "%";
const money = v => v == null ? "–" : "$" + Math.round(v).toLocaleString();
const num = (v, d = 2) => v == null ? "–" : v.toFixed(d);
const signCls = v => v == null ? "" : v >= 0 ? "pos" : "neg";

Chart.defaults.color = css("--muted");
Chart.defaults.borderColor = css("--grid");
Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
Chart.defaults.plugins.tooltip.backgroundColor = css("--surface");
Chart.defaults.plugins.tooltip.titleColor = css("--ink");
Chart.defaults.plugins.tooltip.bodyColor = css("--ink-2");
Chart.defaults.plugins.tooltip.borderColor = css("--border");
Chart.defaults.plugins.tooltip.borderWidth = 1;
const ms = v => v == null ? "–" : v >= 10000 ? (v / 1000).toFixed(0) + " s" : v >= 1000 ? (v / 1000).toFixed(1) + " s" : Math.round(v) + " ms";
const bp = (v, d = 1) => v == null ? "–" : v.toFixed(d) + " bp";
