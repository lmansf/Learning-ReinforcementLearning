// Reward Signal notebook glue — kernel bridge shared by the quiz and widget bundles.
// Colab: google.colab.kernel.invokeFunction (callbacks registered by rlcolab.setup()).
// Classic Jupyter: Jupyter.notebook.kernel.execute. Anything else: no bridge (display only).
export function call(name, args) {
  try {
    const g = window.google;
    if (g && g.colab && g.colab.kernel && g.colab.kernel.invokeFunction) {
      return g.colab.kernel.invokeFunction(name, args, {}).then((r) => (r && r.data && r.data["application/json"]) || null);
    }
  } catch (e) { /* fall through */ }
  try {
    const k = window.Jupyter && window.Jupyter.notebook && window.Jupyter.notebook.kernel;
    if (k) {
      k.execute(`import rlcolab; rlcolab._js_call(${JSON.stringify(name)}, ${JSON.stringify(JSON.stringify(args))})`);
      return Promise.resolve(null);
    }
  } catch (e) { /* fall through */ }
  return Promise.reject(new Error("no kernel bridge"));
}
