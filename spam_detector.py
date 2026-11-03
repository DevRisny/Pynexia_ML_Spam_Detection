"""
Spam Detection with NLP - Pynexia Machine Learning Internship, Week 2.

Pipeline
  1. Text preprocessing : lowercase -> mask URLs / phone numbers / money / numbers
                          -> tokenisation -> stop-word removal -> Porter stemming
  2. Features           : TF-IDF on word uni+bigrams (+ character n-grams for SVM)
  3. Models             : Multinomial Naive Bayes and Linear SVM (plus Logistic
                          Regression as a reference), tuned with 5-fold CV
  4. Evaluation         : held-out stratified test set (20%), precision / recall /
                          F1 / ROC-AUC, confusion matrices, error analysis
  5. Output             : saved model (joblib), charts, metrics JSON, a CLI to
                          classify new messages, and a phishing-signal report

Dataset: SMS Spam Collection (UCI), 5,574 real messages (4,827 ham, 747 spam).

Usage
  python spam_detector.py train                 # train, evaluate, save everything
  python spam_detector.py predict "WINNER! Claim your prize now"
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.base import clone
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sms_spam.tsv"
OUT = ROOT / "outputs"
IMG = ROOT / "docs" / "images"
SEED = 42

# Light, professional palette
BLUE, CYAN, INK, MUTED, RED, GREEN, GRID = "#3157ff", "#03b8d9", "#1b2559", "#5b6478", "#f0566a", "#1fb57a", "#e6eaf5"

# --------------------------------------------------------------------------- #
# 1. Preprocessing
# --------------------------------------------------------------------------- #
_stemmer = PorterStemmer()
STOP = set(ENGLISH_STOP_WORDS) - {"free", "call", "now"}  # keep strong spam cues
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|co\.uk|biz)\S*)", re.I)
PHONE_RE = re.compile(r"\b\d{5,}\b")
MONEY_RE = re.compile(r"[£$€]\s?\d+|\d+\s?(p|pence|pounds?|dollars?)\b", re.I)
NUM_RE = re.compile(r"\b\d+\b")
TOKEN_RE = re.compile(r"[a-z]+")


def clean_text(text: str) -> str:
    """Normalise a message into a space-separated string of stemmed tokens."""
    t = text.lower()
    t = URL_RE.sub(" urltoken ", t)
    t = MONEY_RE.sub(" moneytoken ", t)
    t = PHONE_RE.sub(" phonetoken ", t)
    t = NUM_RE.sub(" numtoken ", t)
    tokens = TOKEN_RE.findall(t)                                   # tokenisation
    tokens = [w for w in tokens if w not in STOP and len(w) > 1]   # stop-word removal
    return " ".join(_stemmer.stem(w) for w in tokens)              # stemming


def phishing_signals(text: str) -> list[str]:
    """Human-readable red flags, shown next to every prediction."""
    flags = []
    if URL_RE.search(text):
        flags.append("contains a link")
    if PHONE_RE.search(text):
        flags.append("asks you to call/text a number")
    if MONEY_RE.search(text) or re.search(r"\b(prize|cash|won|winner|reward)\b", text, re.I):
        flags.append("promises money or a prize")
    if re.search(r"\b(urgent|immediately|now|expires?|last chance|act fast)\b", text, re.I):
        flags.append("creates urgency")
    if re.search(r"\b(verify|account|password|bank|pin|login|suspended)\b", text, re.I):
        flags.append("asks about account / security details")
    if sum(c.isupper() for c in text) > 0.3 * max(1, sum(c.isalpha() for c in text)):
        flags.append("lots of CAPITAL letters")
    return flags


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "message"], quoting=3)
    # the corpus stores some characters as HTML entities (&lt;#&gt; is a redacted number)
    df["message"] = df.message.map(lambda m: html.unescape(html.unescape(m)).replace("<#>", "0"))
    df = df.dropna().drop_duplicates(subset="message").reset_index(drop=True)
    df["y"] = (df.label == "spam").astype(int)
    df["clean"] = df.message.map(clean_text)
    return df


# --------------------------------------------------------------------------- #
# 2-3. Models
# --------------------------------------------------------------------------- #
def build_models() -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    nb = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                   ("clf", MultinomialNB())])
    svm = Pipeline([("feats", FeatureUnion([
                        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
                    ])),
                    ("clf", LinearSVC(class_weight="balanced", random_state=SEED))])
    lr = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                   ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    return {
        "Multinomial Naive Bayes": GridSearchCV(nb, {"clf__alpha": [0.01, 0.05, 0.1, 0.3, 1.0]}, scoring="f1", cv=cv, n_jobs=-1),
        "Linear SVM": GridSearchCV(svm, {"clf__C": [0.1, 0.3, 1.0, 3.0]}, scoring="f1", cv=cv, n_jobs=-1),
        "Logistic Regression": GridSearchCV(lr, {"clf__C": [1, 3, 10, 30]}, scoring="f1", cv=cv, n_jobs=-1),
    }


def scores_of(model, X):
    est = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


# --------------------------------------------------------------------------- #
# 4. Charts
# --------------------------------------------------------------------------- #
def _style(ax, title):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=14, fontweight="bold", pad=12)


def save_charts(df, results, y_test, test_scores, nb_model):
    IMG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    # class balance + message length
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df.label.value_counts()
    axes[0].bar(counts.index, counts.values, color=[CYAN, RED], width=0.55)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 60, f"{v:,}", ha="center", color=INK, fontweight="bold")
    _style(axes[0], "Messages per class")
    lens = df.message.str.len()
    axes[1].hist(lens[df.y == 0], bins=60, range=(0, 300), alpha=0.75, color=CYAN, label="ham")
    axes[1].hist(lens[df.y == 1], bins=60, range=(0, 300), alpha=0.85, color=RED, label="spam")
    axes[1].legend(frameon=False)
    axes[1].set_xlabel("characters", color=MUTED)
    _style(axes[1], "Spam messages are longer")
    fig.tight_layout(); fig.savefig(IMG / "eda.png", dpi=160); plt.close(fig)

    # model comparison
    names = list(results)
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k, (m, c) in enumerate(zip(metrics, [BLUE, CYAN, GREEN])):
        vals = [results[n][m] * 100 for n in names]
        b = ax.bar(x + (k - 1) * w, vals, w, color=c, label=m.capitalize())
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, names); ax.set_ylim(80, 104)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_ylabel("% on held-out test set", color=MUTED)
    ax.set_title("Model comparison (spam class)", loc="left", color=INK, fontsize=14, fontweight="bold", pad=34)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.tight_layout(); fig.savefig(IMG / "model_comparison.png", dpi=160); plt.close(fig)

    # confusion matrices
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#eef2ff", BLUE])
    for ax, n in zip(axes, names):
        cm = np.array(results[n]["confusion_matrix"])
        ax.imshow(cm, cmap=cmap)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=16, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["pred ham", "pred spam"]); ax.set_yticks([0, 1], ["ham", "spam"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, colors=MUTED)
        ax.set_title(n, color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(IMG / "confusion_matrices.png", dpi=160); plt.close(fig)

    # precision-recall curves
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for n, c in zip(names, [BLUE, CYAN, GREEN]):
        p, r, _ = precision_recall_curve(y_test, test_scores[n])
        ax.plot(r, p, color=c, lw=2.2, label=n)
    ax.set_xlabel("Recall", color=MUTED); ax.set_ylabel("Precision", color=MUTED)
    ax.set_xlim(0, 1.01); ax.set_ylim(0.5, 1.01); ax.legend(frameon=False, loc="lower left")
    _style(ax, "Precision–recall curves")
    fig.tight_layout(); fig.savefig(IMG / "pr_curves.png", dpi=160); plt.close(fig)

    # top spam / ham words (Naive Bayes log-prob ratio, readable for everyone)
    nb = nb_model
    vec, clf = nb.named_steps["tfidf"], nb.named_steps["clf"]
    vocab = np.array(vec.get_feature_names_out())
    ratio = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    top_s, top_h = np.argsort(ratio)[-15:], np.argsort(ratio)[:15]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(vocab[top_s], ratio[top_s], color=RED); _style(axes[0], "Strongest SPAM signals")
    axes[1].barh(vocab[top_h][::-1], -ratio[top_h][::-1], color=GREEN); _style(axes[1], "Strongest HAM signals")
    for ax in axes:
        ax.set_xlabel("log-probability ratio", color=MUTED)
    fig.tight_layout(); fig.savefig(IMG / "top_words.png", dpi=160); plt.close(fig)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def train() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df):,} unique messages  (ham {int((df.y == 0).sum()):,} | spam {int(df.y.sum()):,})")
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        df.clean, df.y, df.message, test_size=0.2, stratify=df.y, random_state=SEED)

    results, test_scores, fitted = {}, {}, {}
    for name, search in build_models().items():
        search.fit(X_tr, y_tr)
        pred = search.predict(X_te)
        s = scores_of(search, X_te)
        test_scores[name] = s
        fitted[name] = search.best_estimator_
        results[name] = {
            "best_params": search.best_params_,
            "cv_f1": round(search.best_score_, 4),
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, s),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
        r = results[name]
        print(f"{name:<24} acc {r['accuracy']:.4f}  prec {r['precision']:.4f}  rec {r['recall']:.4f}  "
              f"f1 {r['f1']:.4f}  auc {r['roc_auc']:.4f}  {search.best_params_}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest model on test F1: {best_name}")

    # error analysis for the best model
    pred = fitted[best_name].predict(X_te)
    errors = pd.DataFrame({"message": m_te.values, "actual": np.where(y_te == 1, "spam", "ham"),
                           "predicted": np.where(pred == 1, "spam", "ham")})
    errors = errors[errors.actual != errors.predicted]
    errors.to_csv(OUT / "misclassified_examples.csv", index=False)

    save_charts(df, results, y_te, test_scores, fitted["Multinomial Naive Bayes"])

    # final model: best pipeline refitted on all data + Platt scaling, so the raw
    # SVM margin becomes a spam probability. Platt parameters are learned on
    # out-of-fold margins, so they are not fitted on the model's own training data.
    final = clone(fitted[best_name])   # fresh copy: the train-split model stays untouched
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    if hasattr(final, "predict_proba"):
        platt = None
        final.fit(df.clean, df.y)
    else:
        margins = cross_val_predict(final, df.clean, df.y, cv=cv, method="decision_function")
        lr = LogisticRegression(C=1e6).fit(margins.reshape(-1, 1), df.y)
        platt = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        final.fit(df.clean, df.y)
    joblib.dump({"model": final, "name": best_name, "platt": platt}, OUT / "spam_model.joblib")

    # held-out test probabilities for the live threshold explorer: margins from the
    # model trained on the training split, Platt-scaled with out-of-fold margins
    # from that same training split (the test set is never used for fitting)
    best_cv = fitted[best_name]
    if hasattr(best_cv, "predict_proba"):
        test_p = best_cv.predict_proba(X_te)[:, 1]
    else:
        tr_m = cross_val_predict(best_cv, X_tr, y_tr, cv=cv, method="decision_function")
        pl = LogisticRegression(C=1e6).fit(tr_m.reshape(-1, 1), y_tr)
        test_p = pl.predict_proba(best_cv.decision_function(X_te).reshape(-1, 1))[:, 1]
    test_payload = {"y": [int(v) for v in y_te], "p": [round(float(v), 4) for v in test_p]}
    all_results = {n: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "best_params"}
                   for n, r in results.items()}
    export_web_model(final, platt, best_name, df, summary_metrics=results[best_name],
                     test=test_payload, all_results=all_results)

    summary = {"dataset": "SMS Spam Collection (UCI)", "messages": int(len(df)),
               "ham": int((df.y == 0).sum()), "spam": int(df.y.sum()),
               "test_size": int(len(y_te)), "best_model": best_name,
               "results": {k: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in r.items()}
                           for k, r in results.items()},
               "misclassified_in_test": int(len(errors))}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved model, metrics, {len(errors)} misclassified examples and charts.")


def spam_probability(bundle: dict, cleaned: list[str]) -> np.ndarray:
    model, platt = bundle["model"], bundle.get("platt")
    if platt is None:
        return model.predict_proba(cleaned)[:, 1]
    a, b = platt
    return 1.0 / (1.0 + np.exp(-(a * model.decision_function(cleaned) + b)))


def export_web_model(model, platt, name, df, summary_metrics, test=None, all_results=None) -> None:
    """Export the trained SVM (TF-IDF vocabularies, IDF weights, coefficients,
    Platt scaling) to JSON so the live demo runs the *same* model in the browser."""
    union = model.named_steps["feats"]
    word, char = union.transformer_list[0][1], union.transformer_list[1][1]
    coef = model.named_steps["clf"].coef_.ravel()
    nw = len(word.vocabulary_)

    def table(vec, offset):
        return {t: [round(float(vec.idf_[i]), 5), round(float(coef[offset + i]), 5)]
                for t, i in vec.vocabulary_.items()}

    # word -> stem map for every word seen in the data, so the browser reproduces
    # the Porter stemmer exactly for known words
    raw_words = set(TOKEN_RE.findall(" ".join(df.message.str.lower())))
    raw_words |= {"urltoken", "moneytoken", "phonetoken", "numtoken"}
    stems = {w: _stemmer.stem(w) for w in sorted(raw_words) if w not in STOP and len(w) > 1}
    payload = {
        "model": name,
        "intercept": round(float(model.named_steps["clf"].intercept_[0]), 6),
        "platt": [round(platt[0], 6), round(platt[1], 6)] if platt else None,
        "stop": sorted(STOP),
        "stems": {w: s for w, s in stems.items() if w != s},
        "word": table(word, 0),
        "char": table(char, nw),
        "metrics": {k: round(summary_metrics[k], 4) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        "trained_on": int(len(df)),
        "spam_count": int(df.y.sum()),
        "test": test,
        "results": all_results,
    }
    path = ROOT / "docs" / "model.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Exported browser model -> {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


def predict(messages: list[str]) -> list[dict]:
    bundle = joblib.load(OUT / "spam_model.joblib")
    probs = spam_probability(bundle, [clean_text(m) for m in messages])
    out = []
    for m, p in zip(messages, probs):
        label = "SPAM" if p >= 0.5 else "HAM"
        flags = phishing_signals(m)
        out.append({"message": m, "label": label, "spam_probability": round(float(p), 4), "red_flags": flags})
        print(f"\n[{label}] spam probability {p:.1%}\n  {m}")
        if flags:
            print("  red flags: " + "; ".join(flags))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Spam detector (TF-IDF + Naive Bayes / Linear SVM)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="train, evaluate and save the model")
    p = sub.add_parser("predict", help="classify one or more messages")
    p.add_argument("messages", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "train":
        train()
    else:
        predict(args.messages)


if __name__ == "__main__":
    main(sys.argv[1:])
