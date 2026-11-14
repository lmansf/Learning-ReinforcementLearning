// Reward Signal notebook glue — knowledge checks (same markup and behaviour as the app's renderQuiz).
import { renderMarkdown } from "./markdown.js";
import { call } from "./nb-bridge.js";

const inlineMd = (s) => renderMarkdown(String(s)).html.replace(/^<p>|<\/p>$/g, "");

export function quiz(id, q, key, st) {
  const node = document.getElementById(id);
  if (!node) return;
  st = st || {};
  const head = (txt, warn) => { const el = node.querySelector(".quiz-head .label.dim"); el.textContent = txt; el.style.color = warn ? "var(--amber)" : ""; };
  node.className = "quiz";
  node.innerHTML = `
    <div class="quiz-head"><span class="label">KNOWLEDGE CHECK</span><span class="label dim">${st.state ? (st.state === "right" ? "+25 XP EARNED" : "ANSWERED") : "+25 XP"}</span></div>
    <div class="quiz-q">${inlineMd(q.q)}</div>
    <div class="quiz-opts">${q.options.map((o, i) => `<button class="quiz-opt" data-i="${i}"><span class="k">${String.fromCharCode(65 + i)}</span><span>${inlineMd(o)}</span></button>`).join("")}</div>
    <div class="quiz-explain" hidden></div>`;
  const opts = [...node.querySelectorAll(".quiz-opt")];
  const reveal = (picked) => {
    opts.forEach((b, i) => { b.disabled = true; if (i === q.answer) b.classList.add("right"); else if (i === picked) b.classList.add("wrong"); });
    const ex = node.querySelector(".quiz-explain");
    ex.hidden = !q.explain && picked === q.answer;
    ex.className = "quiz-explain" + (picked === q.answer ? "" : " miss");
    ex.innerHTML = (picked === q.answer ? "<strong>Correct.</strong> " : "<strong>Not quite.</strong> ") + (q.explain ? inlineMd(q.explain) : "");
  };
  if (st.state) reveal(st.state === "right" ? q.answer : (st.pick ?? -1));
  opts.forEach((b) => b.onclick = () => {
    const i = +b.dataset.i, right = i === q.answer;
    reveal(i);
    head(right ? "+25 XP EARNED" : "ANSWERED");
    call("rs.quiz", [key, i, right]).then((r) => {
      if (r && r.xp != null) head(`${r.awarded ? "+25 XP EARNED" : right ? "CORRECT" : "ANSWERED"} · ${r.xp} XP · ${r.rank}`);
    }).catch(() => head("ANSWER NOT SAVED · RUN THE SETUP CELL FIRST", true));
  });
  // retry: double-click the header (no extra XP)
  node.querySelector(".quiz-head").ondblclick = () => quiz(id, q, key, {});
}
