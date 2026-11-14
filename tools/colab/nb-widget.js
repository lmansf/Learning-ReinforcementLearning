// Reward Signal notebook glue — the app's interactive lesson widgets (app/js/widgets.js).
// Core Runner levels come inlined (no server): levels 1–3 as data, level 4 from pre-generated wrecks.
import { mountWidget } from "./widgets.js";
import { call } from "./nb-bridge.js";

export function widget(id, opts, data) {
  const host = document.getElementById(id);
  if (!host) return;
  const l4 = data.l4 || [];
  const api = async (path) => {
    const m = /seed=(\d+)/.exec(path);
    if (m) return { map: l4[(+m[1] - 1) % Math.max(1, l4.length)] };
    return { levels: data.levels };
  };
  const mem = Object.assign({}, data.store || {});
  const store = {
    get: (k, d) => (k in mem ? mem[k] : d),
    set: (k, v) => { mem[k] = v; call("rs.store", [k, v]).catch(() => {}); },
  };
  mountWidget(host, opts, { api, store });
}
