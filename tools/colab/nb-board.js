// Reward Signal notebook glue — rlhud visuals: one VisualBoard (app/js/visuals.js) per output.
import { VisualBoard } from "./visuals.js";

export function board(id, events) {
  const host = document.getElementById(id);
  if (!host) return;
  host.classList.add("visuals", "rs-board");
  const b = new VisualBoard(host);
  for (const ev of events) {
    try { b.handle(ev); } catch (e) { console.error(e); }
  }
}
