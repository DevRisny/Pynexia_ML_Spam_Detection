/* SpamShield page logic: inspector, live inbox, pipeline trace, threshold explorer. */
(async function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pctText = (p) => { const v = p * 100; if (v >= 99.95) return ">99.9%"; if (v < 0.05) return "<0.1%"; return (v < 1 || v > 99 ? v.toFixed(1) : Math.round(v)) + "%"; };
  const fmt = (x, d = 1) => (x * 100).toFixed(d) + "%";

  // sticky nav border
  const nav = $("nav");
  addEventListener("scroll", () => nav.classList.toggle("scrolled", scrollY > 8), { passive: true });

  const ICONS = {
    link: '<path d="M10 14a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1"/><path d="M14 10a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1"/>',
    phone: '<path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a2 2 0 0 1-2 2A16 16 0 0 1 3 6a2 2 0 0 1 2-2"/>',
    gift: '<rect x="3" y="8" width="18" height="4"/><path d="M12 8v13M5 12v9h14v-9M12 8S10 3 7.5 4.5 9 8 12 8Zm0 0s2-5 4.5-3.5S15 8 12 8Z"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    lock: '<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    caps: '<path d="M4 20 9 4h1l5 16M6 14h7M16 20l3-9 3 9M17 17h4"/>',
    ok: '<path d="m5 12 4.5 4.5L19 7"/>',
  };
  const svgIcon = (k) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[k]}</svg>`;

  // ---------------------------------------------------------------- load model
  let engine;
  try {
    const res = await fetch("model.json");
    if (!res.ok) throw new Error(res.status);
    engine = SpamEngine.create(await res.json());
  } catch (e) {
    $("marked").innerHTML = '<span class="muted">The model could not be loaded. Refresh the page to try again.</span>';
    $("status").innerHTML = '<i style="background:var(--spam);box-shadow:0 0 0 4px var(--spam-wash)"></i>Model unavailable';
    return;
  }
  const M = engine.metrics;
  $("k-acc").textContent = fmt(M.accuracy);
  $("k-f1").textContent = fmt(M.f1);
  $("k-auc").textContent = M.roc_auc.toFixed(3);
  $("k-n").textContent = engine.trainedOn.toLocaleString();

  // ---------------------------------------------------------------- inspector
  const ARC = 314.16;
  let scanTimer;
  function analyse(text, { animate = true } = {}) {
    if (!text.trim()) {
      $("pct").textContent = "–";
      $("arc").style.strokeDashoffset = ARC; $("arc").setAttribute("stroke", "#c9d4e6");
      $("vtitle").className = "idle"; $("vtitle").textContent = "Waiting for a message";
      $("vsub").textContent = "Type or paste a message, or load an example.";
      $("flags").innerHTML = "";
      $("marked").innerHTML = '<span class="muted">The words that pushed the decision will be highlighted here.</span>';
      renderPipeline("");
      return;
    }
    if (animate) {
      const s = $("scan"); s.classList.remove("run"); void s.offsetWidth; s.classList.add("run");
      clearTimeout(scanTimer); scanTimer = setTimeout(() => s.classList.remove("run"), 750);
    }
    const r = engine.explain(text);
    const p = r.probability;
    $("pct").textContent = pctText(p);
    $("arc").style.strokeDashoffset = ARC * (1 - Math.max(0.01, p));
    $("arc").setAttribute("stroke", p >= 0.5 ? "var(--spam)" : p >= 0.2 ? "var(--amber)" : "var(--safe)");
    const spam = r.label === "spam";
    $("vtitle").className = spam ? "spam" : "ham";
    $("vtitle").textContent = spam ? (p > 0.9 ? "Spam. Do not reply." : "Probably spam") : (p < 0.1 ? "Looks safe" : "Probably safe");
    $("vsub").textContent = spam
      ? "This message matches patterns seen in spam and phishing."
      : "This reads like a normal personal message.";

    const flags = engine.redFlags(text);
    $("flags").innerHTML = flags.length
      ? flags.map((f) => `<span class="flag">${svgIcon(f.icon)}${esc(f.text)}</span>`).join("")
      : `<span class="flag ok">${svgIcon("ok")}No phishing tricks found</span>`;

    const max = Math.max(0.04, ...r.pieces.map((x) => Math.abs(x.score)));
    $("marked").innerHTML = r.pieces.map((x) => {
      const a = Math.abs(x.score) / max;
      if (!/\S/.test(x.text) || a < 0.12) return esc(x.text);
      const lvl = a > 0.66 ? 3 : a > 0.33 ? 2 : 1;
      const dir = x.score > 0 ? "s" : "h";
      return `<mark class="${dir}${lvl}" title="${x.score > 0 ? "Pushes towards spam" : "Pushes towards safe"} (${x.score > 0 ? "+" : ""}${x.score.toFixed(2)})">${esc(x.text)}</mark>`;
    }).join("");
    renderPipeline(text);
  }

  const msg = $("msg");
  let debounce;
  msg.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(() => analyse(msg.value), 120); });
  document.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => { msg.value = b.dataset.s; analyse(msg.value); }));

  // ---------------------------------------------------------------- pipeline
  function renderPipeline(text) {
    const ids = ["p-mask", "p-tok", "p-stem", "p-feat", "p-svm"];
    if (!text.trim()) { ids.forEach((i) => ($(i).innerHTML = '<span class="muted">Waiting for a message…</span>')); return; }
    const t = engine.trace(text);
    $("p-mask").innerHTML = esc(t.masked).replace(/\b(urltoken|phonetoken|moneytoken|numtoken)\b/g, '<span class="tok special">$1</span>');
    const keptSet = new Set(t.stemmed.map((s) => s.from));
    $("p-tok").innerHTML = t.raw.map((w) => {
      const special = /^(urltoken|phonetoken|moneytoken|numtoken)$/.test(w);
      return `<span class="tok ${keptSet.has(w) ? (special ? "special" : "") : "gone"}">${esc(w)}</span>`;
    }).join("") + `<div class="muted" style="margin-top:6px">${t.removed.length} removed, ${t.stemmed.length} kept</div>`;
    const changed = t.stemmed.filter((s) => s.from !== s.to);
    $("p-stem").innerHTML = (changed.length
      ? changed.map((s) => `<div class="stemrow"><span>${esc(s.from)}</span><b>→ ${esc(s.to)}</b></div>`).join("")
      : '<span class="muted">No words needed stemming.</span>') +
      `<div class="muted" style="margin-top:6px">${t.stemmed.length - changed.length} already at their root</div>`;
    const top = t.features.slice(0, 7);
    const fmax = Math.max(1e-6, ...top.map((f) => Math.abs(f.x)));
    $("p-feat").innerHTML = top.map((f) => {
      const w = (Math.abs(f.x) / fmax) * 50;
      const style = f.x > 0 ? `left:50%;width:${w}%;background:var(--spam)` : `right:50%;width:${w}%;background:var(--safe)`;
      const label = f.kind === "char" ? `"${f.f.replace(/ /g, "␣")}"` : f.f;
      return `<div class="fbar"><span class="lab">${esc(label)}<em>${f.kind === "char" ? "letters" : "word"}</em></span><span class="bar"><i style="${style}"></i></span></div>`;
    }).join("") + `<div class="muted" style="margin-top:6px">${t.nWord} word + ${t.nChar} letter features active</div>`;
    const sign = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);
    $("p-svm").innerHTML = `
      <div><span>Word features</span><span class="num">${sign(t.wordDot)}</span></div>
      <div><span>Letter features</span><span class="num">${sign(t.charDot)}</span></div>
      <div><span>Bias</span><span class="num">${sign(t.intercept)}</span></div>
      <div class="total"><span>SVM score</span><span class="num">${sign(t.margin)}</span></div>
      <div class="final num" style="color:${t.probability >= 0.5 ? "var(--spam-deep)" : "var(--safe-deep)"}">${pctText(t.probability)} spam</div>`;
  }

  // ---------------------------------------------------------------- inbox
  const MAILS = [
    { from: "Amaya", av: "#6d5efc", time: "9:41", text: "Are you coming to the study group later? We're starting chapter 4 at the library." },
    { from: "+44 7781 902 113", av: "#e5484d", time: "9:30", text: "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only." },
    { from: "Dad", av: "#12a594", time: "8:52", text: "Reached the shop safely. Will call you after lunch, don't forget to charge your phone." },
    { from: "SecureBank Alert", av: "#f5a524", time: "8:15", text: "URGENT: your account is suspended. Verify your PIN now at www.securebank-verify-login.com or your card will be blocked." },
    { from: "Course Team", av: "#2f6feb", time: "Yesterday", text: "Reminder: the project presentation is on Monday at 10am in Lab 2. Please bring your slides on a USB drive." },
    { from: "88600", av: "#e5484d", time: "Yesterday", text: "FREE MSG: You have won a FREE Nokia phone + camera! Txt NOKIA to 88600 now. Only 150p/msg. Stop? txt STOP" },
    { from: "Nimal", av: "#0b7a6d", time: "Mon", text: "Sorry I missed your call, I was in a meeting. What time works for you tomorrow?" },
    { from: "Delivery Update", av: "#f5a524", time: "Mon", text: "Your parcel could not be delivered. Pay the £1.99 redelivery fee within 24 hours: http://parcel-redeliver.co/track" },
  ];
  const initials = (s) => (/^[+\d]/.test(s) ? "#" : s.split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase());
  $("mails").innerHTML = MAILS.map((m, i) => `
    <button class="mail" type="button" data-i="${i}">
      <span class="avatar" style="background:${m.av}">${esc(initials(m.from))}</span>
      <span style="min-width:0"><span class="who">${esc(m.from)}<small>${esc(m.time)}</small></span><span class="preview">${esc(m.text)}</span></span>
      <span class="tag wait" id="tag-${i}">Not checked</span>
    </button>`).join("");
  $("inbox-count").textContent = MAILS.length + " messages";
  document.querySelectorAll(".mail").forEach((el) => el.addEventListener("click", () => {
    msg.value = MAILS[+el.dataset.i].text; analyse(msg.value);
    $("inspector").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  }));

  let scanning = false;
  async function scanInbox() {
    if (scanning) return; scanning = true;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let ns = 0, nh = 0; $("n-spam").textContent = 0; $("n-ham").textContent = 0;
    const rows = [...document.querySelectorAll(".mail")];
    rows.forEach((r, i) => { r.classList.remove("is-spam"); const t = $("tag-" + i); t.className = "tag wait"; t.textContent = "Not checked"; });
    for (let i = 0; i < MAILS.length; i++) {
      const row = rows[i]; const tag = $("tag-" + i);
      tag.textContent = "Checking…"; row.classList.add("scanning");
      if (!reduce) await new Promise((r) => setTimeout(r, 380));
      row.classList.remove("scanning");
      const p = engine.classify(MAILS[i].text).probability;
      const spam = p >= 0.5;
      tag.className = "tag " + (spam ? "spam" : "ham");
      tag.textContent = (spam ? "Spam " : "Safe ") + pctText(spam ? p : 1 - p);
      row.classList.toggle("is-spam", spam);
      spam ? ns++ : nh++;
      $("n-spam").textContent = ns; $("n-ham").textContent = nh;
    }
    scanning = false;
  }
  $("rescan").addEventListener("click", scanInbox);
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); scanInbox(); } }, { threshold: 0.35 });
  io.observe($("inbox"));

  // ---------------------------------------------------------------- leaderboard + dataset
  const R = engine.results || {};
  const order = Object.keys(R).sort((a, b) => R[b].f1 - R[a].f1);
  const COLORS = { precision: "var(--blue)", recall: "#03b8d9", f1: "var(--safe)" };
  $("board").innerHTML = order.map((name, i) => `
    <div class="model">
      <div class="model-head"><span>${esc(name)}${i === 0 ? "<small>chosen</small>" : ""}</span><span class="num">F1 ${fmt(R[name].f1)}</span></div>
      <div class="bars">${["precision", "recall", "f1"].map((k) => `
        <div class="mb"><span>${k === "f1" ? "F1" : k[0].toUpperCase() + k.slice(1)}</span><span class="bar"><i style="width:${((R[name][k] - 0.85) / 0.15) * 100}%;background:${COLORS[k]}"></i></span><span class="num">${fmt(R[name][k])}</span></div>`).join("")}
      </div>
    </div>`).join("") + '<p class="muted" style="font-size:12.5px;margin:0">Bars start at 85% so the differences are visible.</p>';
  const spamN = engine.spamCount, total = engine.trainedOn, frac = spamN / total;
  $("d-spam").textContent = spamN.toLocaleString(); $("d-ham").textContent = (total - spamN).toLocaleString();
  const C = 2 * Math.PI * 15.9;
  $("donut").innerHTML = `<circle cx="21" cy="21" r="15.9" fill="none" stroke="var(--safe-wash)" stroke-width="7"/>
    <circle cx="21" cy="21" r="15.9" fill="none" stroke="var(--spam)" stroke-width="7" stroke-dasharray="${C * frac} ${C}" transform="rotate(-90 21 21)"/>
    <text x="21" y="23.5" text-anchor="middle" font-size="8" font-weight="800" fill="var(--ink)" font-family="Bricolage Grotesque, sans-serif">${Math.round(frac * 100)}%</text>`;

  // ---------------------------------------------------------------- threshold explorer + PR curve
  const T = engine.test;
  function confusion(th) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < T.y.length; i++) {
      const s = T.p[i] >= th;
      if (s && T.y[i]) tp++; else if (s) fp++; else if (T.y[i]) fn++; else tn++;
    }
    const precision = tp + fp ? tp / (tp + fp) : 1, recall = tp + fn ? tp / (tp + fn) : 0;
    return { tp, fp, tn, fn, precision, recall, f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0, acc: (tp + tn) / T.y.length };
  }
  // PR curve over all thresholds
  const W = 360, H = 250, pad = { l: 40, r: 12, t: 12, b: 34 };
  const xs = (r) => pad.l + r * (W - pad.l - pad.r);
  const yLo = 0.5;
  const ys = (p) => pad.t + (1 - (p - yLo) / (1 - yLo)) * (H - pad.t - pad.b);
  const ths = [...new Set(T.p)].sort((a, b) => b - a);
  const pts = ths.map((th) => { const c = confusion(th); return [c.recall, Math.max(yLo, c.precision)]; });
  const path = "M" + pts.map(([r, p]) => `${xs(r).toFixed(1)},${ys(p).toFixed(1)}`).join("L");
  const grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1].map((p) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${ys(p)}" y2="${ys(p)}" stroke="#edf1f7"/><text x="${pad.l - 6}" y="${ys(p) + 4}" text-anchor="end">${p.toFixed(1)}</text>`).join("")
    + [0, 0.25, 0.5, 0.75, 1].map((r) => `<text x="${xs(r)}" y="${H - pad.b + 16}" text-anchor="middle">${r}</text>`).join("");
  $("pr").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision-recall curve with the current threshold marked">
    ${grid}
    <text x="${(pad.l + W - pad.r) / 2}" y="${H - 4}" text-anchor="middle">Recall</text>
    <text x="12" y="${(pad.t + H - pad.b) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(pad.t + H - pad.b) / 2})">Precision</text>
    <path d="${path}" fill="none" stroke="var(--blue)" stroke-width="2.5" stroke-linejoin="round"/>
    <line id="pr-vx" stroke="var(--blue)" stroke-dasharray="3 3" opacity=".5"/>
    <circle id="pr-dot" r="7" fill="#fff" stroke="var(--blue)" stroke-width="3"/>
  </svg>`;

  function updateThreshold() {
    const th = +$("thr").value;
    $("thr-out").textContent = Math.round(th * 100) + "%";
    const c = confusion(th);
    $("t-acc").textContent = fmt(c.acc); $("t-p").textContent = fmt(c.precision); $("t-r").textContent = fmt(c.recall); $("t-f").textContent = fmt(c.f1);
    $("c-tn").textContent = c.tn; $("c-fp").textContent = c.fp; $("c-fn").textContent = c.fn; $("c-tp").textContent = c.tp;
    const dot = $("pr-dot"), vx = $("pr-vx");
    const cx = xs(c.recall), cy = ys(Math.max(yLo, c.precision));
    dot.setAttribute("cx", cx); dot.setAttribute("cy", cy);
    vx.setAttribute("x1", cx); vx.setAttribute("x2", cx); vx.setAttribute("y1", cy); vx.setAttribute("y2", H - pad.b);
  }
  $("thr").addEventListener("input", updateThreshold);
  updateThreshold();

  // ---------------------------------------------------------------- start
  msg.value = document.querySelector(".chip").dataset.s;
  analyse(msg.value, { animate: false });
})();
