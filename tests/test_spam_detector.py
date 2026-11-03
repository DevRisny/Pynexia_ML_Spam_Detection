"""Tests for the SpamShield pipeline. Run with:  python -m pytest -q"""
import json
import subprocess
from pathlib import Path

import pytest

from spam_detector import OUT, clean_text, phishing_signals, predict

ROOT = Path(__file__).resolve().parents[1]


def test_masking_and_stemming():
    out = clean_text("WIN £500 cash now! Call 09061701461 or visit www.prize-claim.com")
    assert "moneytoken" in out and "phonetoken" in out and "urltoken" in out
    assert "win" in out.split() and "cash" in out.split()


def test_stop_words_removed_but_spam_cues_kept():
    toks = clean_text("this is free and you should call now").split()
    assert "this" not in toks and "is" not in toks
    assert {"free", "call", "now"} <= set(toks)


def test_phishing_signals():
    flags = phishing_signals("URGENT: verify your bank account at www.x-login.com")
    assert "contains a link" in flags
    assert "creates urgency" in flags
    assert "asks about account / security details" in flags
    assert phishing_signals("see you at lunch") == []


@pytest.mark.skipif(not (OUT / "spam_model.joblib").exists(), reason="run `python spam_detector.py train` first")
def test_predictions_make_sense():
    res = predict(["Congratulations! You WON a £1000 prize. Text CLAIM to 87121",
                   "Are we still meeting at the library at 5?"])
    assert res[0]["label"] == "SPAM" and res[0]["spam_probability"] > 0.9
    assert res[1]["label"] == "HAM" and res[1]["spam_probability"] < 0.1


@pytest.mark.skipif(not (OUT / "metrics.json").exists(), reason="run `python spam_detector.py train` first")
def test_reported_quality():
    m = json.loads((OUT / "metrics.json").read_text())
    best = m["results"][m["best_model"]]
    assert best["accuracy"] > 0.98 and best["f1"] > 0.93


def test_browser_engine_matches_python():
    """The JavaScript engine used on the website must give the same answers."""
    model = ROOT / "docs" / "model.json"
    if not model.exists():
        pytest.skip("run training first")
    node = subprocess.run(["node", "--version"], capture_output=True)
    if node.returncode != 0:
        pytest.skip("node not installed")
    script = """
      const E=require(process.argv[1]); const m=require(process.argv[2]); const e=E.create(m);
      const s=['WINNER!! Claim your £900 prize, call 09061701461','ok see you at home later'];
      console.log(JSON.stringify(s.map(x=>e.classify(x).probability)));"""
    out = subprocess.run(["node", "-e", script, str(ROOT / "docs" / "spam-engine.js"), str(model)],
                         capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    py = [r["spam_probability"] for r in predict(['WINNER!! Claim your £900 prize, call 09061701461',
                                                    'ok see you at home later'])]
    for a, b in zip(js, py):
        assert abs(a - b) < 0.01
