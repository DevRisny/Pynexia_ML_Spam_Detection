/*
 * SpamShield browser engine.
 * Re-implements the Python pipeline in spam_detector.py so the live demo runs
 * the exact trained Linear SVM: masking -> tokenising -> stop words -> Porter
 * stems -> word (1-2 gram) + char_wb (3-5 gram) TF-IDF -> SVM -> Platt scaling.
 * Works in the browser (window.SpamEngine) and in Node (module.exports).
 */
(function (root) {
  "use strict";

  const URL_RE = /(https?:\/\/\S+|www\.\S+|\b\S+\.(com|net|org|co\.uk|biz)\S*)/gi;
  const PHONE_RE = /\b\d{5,}\b/g;
  const MONEY_RE = /[£$€]\s?\d+|\d+\s?(p|pence|pounds?|dollars?)\b/gi;
  const NUM_RE = /\b\d+\b/g;
  const TOKEN_RE = /[a-z]+/g;

  function create(model) {
    const stop = new Set(model.stop);
    const stems = model.stems;

    // Fallback for words never seen in training: a light Porter step-1 stem.
    // Unseen words carry no word-level weight anyway; this only affects char n-grams.
    function fallbackStem(w) {
      if (w.endsWith("sses")) return w.slice(0, -2);
      if (w.endsWith("ies")) return w.slice(0, -2);
      if (w.endsWith("s") && !w.endsWith("ss") && w.length > 3) w = w.slice(0, -1);
      if (w.endsWith("ing") && /[aeiouy]/.test(w.slice(0, -3)) && w.length > 5) return w.slice(0, -3);
      if (w.endsWith("ed") && /[aeiouy]/.test(w.slice(0, -2)) && w.length > 4) return w.slice(0, -2);
      return w;
    }
    const stem = (w) => (Object.prototype.hasOwnProperty.call(stems, w) ? stems[w] : (model.word[w] ? w : fallbackStem(w)));

    function masked(text) {
      return text.toLowerCase()
        .replace(URL_RE, " urltoken ")
        .replace(MONEY_RE, " moneytoken ")
        .replace(PHONE_RE, " phonetoken ")
        .replace(NUM_RE, " numtoken ");
    }

    /** Same output as clean_text() in Python: list of stemmed tokens. */
    function tokens(text) {
      const raw = masked(text).match(TOKEN_RE) || [];
      return raw.filter((w) => !stop.has(w) && w.length > 1).map(stem);
    }

    function wordGrams(toks) {
      const out = toks.slice();
      for (let i = 0; i + 1 < toks.length; i++) out.push(toks[i] + " " + toks[i + 1]);
      return out;
    }

    // Mirrors sklearn's _char_wb_ngrams for ngram_range (3, 5)
    function charGrams(clean) {
      const out = [];
      for (const word of clean.split(/\s+/).filter(Boolean)) {
        const w = " " + word + " ";
        for (let n = 3; n <= 5; n++) {
          let offset = 0;
          out.push({ g: w.slice(offset, offset + n), word });
          while (offset + n < w.length) {
            offset += 1;
            out.push({ g: w.slice(offset, offset + n), word });
          }
          if (offset === 0) break;
        }
      }
      return out;
    }

    // sublinear TF-IDF + L2 norm, returns {weights: Map(feature -> value*coef), dot}
    function block(grams, table, keyOf) {
      const counts = new Map();
      for (const g of grams) {
        const k = keyOf(g);
        if (table[k]) counts.set(k, (counts.get(k) || 0) + 1);
      }
      let norm = 0;
      const vals = new Map();
      for (const [k, c] of counts) {
        const v = (1 + Math.log(c)) * table[k][0];
        vals.set(k, v);
        norm += v * v;
      }
      norm = Math.sqrt(norm) || 1;
      let dot = 0;
      const contrib = new Map();
      for (const [k, v] of vals) {
        const x = (v / norm) * table[k][1];
        contrib.set(k, x);
        dot += x;
      }
      return { dot, contrib };
    }

    function classify(text) {
      const toks = tokens(text);
      const clean = toks.join(" ");
      const wb = block(wordGrams(toks), model.word, (g) => g);
      const cgrams = charGrams(clean);
      const cb = block(cgrams, model.char, (g) => g.g);
      const margin = wb.dot + cb.dot + model.intercept;
      const p = model.platt ? 1 / (1 + Math.exp(-(model.platt[0] * margin + model.platt[1]))) : 1 / (1 + Math.exp(-margin));

      // Attribute every feature's contribution to the stemmed token(s) it came from
      const perStem = new Map();
      const add = (s, x) => perStem.set(s, (perStem.get(s) || 0) + x);
      for (const [k, x] of wb.contrib) {
        const parts = k.split(" ");
        for (const s of parts) add(s, x / parts.length);
      }
      // a char n-gram can appear in several words; split its weight by occurrences
      const occ = new Map();
      for (const g of cgrams) if (cb.contrib.has(g.g)) occ.set(g.g, (occ.get(g.g) || 0) + 1);
      const seen = new Map();
      for (const g of cgrams) {
        if (!cb.contrib.has(g.g)) continue;
        const key = g.g + "|" + g.word;
        if (seen.has(key)) continue;
        seen.set(key, true);
        const share = cgrams.filter((h) => h.g === g.g && h.word === g.word).length / occ.get(g.g);
        add(g.word, cb.contrib.get(g.g) * share);
      }
      return { probability: p, margin, label: p >= 0.5 ? "spam" : "ham", tokens: toks, perStem };
    }

    /** Split original text into display pieces, each with its contribution. */
    function explain(text) {
      const result = classify(text);
      const pieces = [];
      const re = /(https?:\/\/\S+|www\.\S+|\S+@\S+|[A-Za-z£$€0-9'.]+|\s+|[^\sA-Za-z0-9]+)/g;
      let m;
      while ((m = re.exec(text)) !== null) {
        const piece = m[0];
        let score = 0;
        if (/\S/.test(piece)) {
          for (const t of tokens(piece)) score += result.perStem.get(t) || 0;
        }
        pieces.push({ text: piece, score });
      }
      return { ...result, pieces };
    }

    function redFlags(text) {
      const f = [];
      const t = text;
      if (new RegExp(URL_RE.source, "i").test(t)) f.push({ icon: "link", text: "Contains a link" });
      if (new RegExp(PHONE_RE.source).test(t)) f.push({ icon: "phone", text: "Asks you to call or text a number" });
      if (new RegExp(MONEY_RE.source, "i").test(t) || /\b(prize|cash|won|winner|reward)\b/i.test(t)) f.push({ icon: "gift", text: "Promises money or a prize" });
      if (/\b(urgent|immediately|now|expires?|last chance|act fast)\b/i.test(t)) f.push({ icon: "clock", text: "Creates urgency" });
      if (/\b(verify|account|password|bank|pin|login|suspended)\b/i.test(t)) f.push({ icon: "lock", text: "Asks about account or security details" });
      const letters = (t.match(/[A-Za-z]/g) || []).length;
      const caps = (t.match(/[A-Z]/g) || []).length;
      if (letters > 8 && caps > 0.3 * letters) f.push({ icon: "caps", text: "Lots of CAPITAL letters" });
      return f;
    }

    /** Every intermediate stage of the pipeline, for the "how it decides" view. */
    function trace(text) {
      const m = masked(text);
      const raw = m.match(TOKEN_RE) || [];
      const kept = raw.filter((w) => !stop.has(w) && w.length > 1);
      const removed = raw.filter((w) => stop.has(w) || w.length <= 1);
      const stemmed = kept.map((w) => ({ from: w, to: stem(w) }));
      const r = classify(text);
      const toks = r.tokens;
      const wb = block(wordGrams(toks), model.word, (g) => g);
      const cb = block(charGrams(toks.join(" ")), model.char, (g) => g.g);
      const feats = [...wb.contrib.entries()].map(([k, x]) => ({ f: k, kind: "word", x }))
        .concat([...cb.contrib.entries()].map(([k, x]) => ({ f: k, kind: "char", x })))
        .sort((a, b) => Math.abs(b.x) - Math.abs(a.x));
      return {
        masked: m.replace(/\s+/g, " ").trim(), raw, removed, stemmed, features: feats,
        nWord: wb.contrib.size, nChar: cb.contrib.size,
        wordDot: wb.dot, charDot: cb.dot, intercept: model.intercept, margin: r.margin, probability: r.probability,
      };
    }

    return { tokens, classify, explain, redFlags, trace, metrics: model.metrics, trainedOn: model.trained_on,
             spamCount: model.spam_count, name: model.model, test: model.test, results: model.results };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SpamEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
