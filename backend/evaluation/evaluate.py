"""Evaluate ISL Bridge recognizers on recorded landmark clips.

Usage (from backend/):
    python -m evaluation.evaluate data/samples/*.jsonl
    python -m evaluation.evaluate data/samples/*.jsonl --out ../runs/eval/my-run

Input: JSONL files exported from the app's "Dataset recorder" tab. One sample
per line with keys: label (sign id or NEGATIVE), signer, lighting, frames.

What it reports
  1. Rule-based recognizer (the deployed one, recognize.py) on every sample.
     It has no training, so every sample is a fair test sample.
  2. Learned baselines on landmark features (evaluation/features.py):
     logistic regression, k-nearest neighbours and random forest, untuned defaults.
     * leave-one-signer-out (LOSO) when there are >= 2 signers: train on all
       other people, test on the held-out person. This is the honest estimate
       of performance on a new user.
     * stratified 5-fold within the pooled data, for reference. This is
       optimistic because the same person appears in train and test.
     Learned models are trained with NEGATIVE as an 11th class and also
     reject any prediction whose probability is below --reject-threshold,
     mirroring the app's UNRECOGNIZED state.

Metrics (UNRECOGNIZED predictions count as "rejected"):
  * sign accuracy      – sign samples predicted as the correct sign
  * wrong-sign rate    – sign samples predicted as a *different* sign
                          (the most harmful error: it speaks the wrong word)
  * rejection rate     – sign samples predicted UNRECOGNIZED
  * false-accept rate  – NEGATIVE samples predicted as any sign
  * macro-F1 over the 10 signs + NEGATIVE (UNRECOGNIZED mapped to NEGATIVE)
  95% Wilson intervals are given for the proportions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catalog import load_catalog  # noqa: E402
from recognize import recognize_frames  # noqa: E402

NEG = "NEGATIVE"
REJ = "UNRECOGNIZED"


# ---------------------------------------------------------------- data

def load_samples(paths: list[str]) -> list[dict]:
    samples, seen = [], set()
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                sid = s.get("id") or f"{p}:{ln}"
                if sid in seen:  # same sample exported twice
                    continue
                seen.add(sid)
                for key in ("label", "frames"):
                    if key not in s:
                        raise ValueError(f"{p}:{ln} missing '{key}'")
                s.setdefault("signer", "S1")
                s.setdefault("lighting", "unknown")
                samples.append(s)
    return samples


# ---------------------------------------------------------------- metrics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(y_true: list[str], y_pred: list[str], sign_ids: list[str]) -> dict:
    sign_idx = [i for i, t in enumerate(y_true) if t != NEG]
    neg_idx = [i for i, t in enumerate(y_true) if t == NEG]

    correct = sum(y_pred[i] == y_true[i] for i in sign_idx)
    rejected = sum(y_pred[i] == REJ for i in sign_idx)
    wrong = len(sign_idx) - correct - rejected
    false_acc = sum(y_pred[i] not in (REJ, NEG) for i in neg_idx)

    classes = sign_ids + [NEG]
    mapped = [NEG if p == REJ else p for p in y_pred]
    f1s, per_class = [], {}
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, mapped))
        fp = sum(t != c and p == c for t, p in zip(y_true, mapped))
        fn = sum(t == c and p != c for t, p in zip(y_true, mapped))
        support = tp + fn
        if support == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_class[c] = {"support": support, "precision": prec, "recall": rec, "f1": f1}

    def prop(k, n):
        lo, hi = wilson(k, n)
        return {"k": k, "n": n, "rate": (k / n) if n else float("nan"), "ci95": [lo, hi]}

    return {
        "sign_accuracy": prop(correct, len(sign_idx)),
        "wrong_sign_rate": prop(wrong, len(sign_idx)),
        "rejection_rate": prop(rejected, len(sign_idx)),
        "false_accept_rate": prop(false_acc, len(neg_idx)),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_class": per_class,
    }


def confusion(y_true, y_pred, rows, cols) -> np.ndarray:
    m = np.zeros((len(rows), len(cols)), dtype=int)
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: i for i, c in enumerate(cols)}
    for t, p in zip(y_true, y_pred):
        m[ri[t], ci[p]] += 1
    return m


# ---------------------------------------------------------------- rule-based

def run_rule_based(samples: list[dict]) -> tuple[list[str], Counter]:
    preds, codes = [], Counter()
    for s in samples:
        r = recognize_frames(s["frames"])
        if r["state"] == "recognized":
            preds.append(r["sign_id"])
        else:
            preds.append(REJ)
            codes[r.get("code", "unknown")] += 1
    return preds, codes


# ---------------------------------------------------------------- learned

def _models(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=seed),
    }


def _predict_with_reject(model, X, threshold: float) -> list[str]:
    proba = model.predict_proba(X)
    classes = model.classes_
    out = []
    for row in proba:
        j = int(np.argmax(row))
        c = classes[j]
        out.append(REJ if (c == NEG or row[j] < threshold) else c)
    return out


def run_learned(samples, X, threshold: float, seed: int):
    from sklearn.model_selection import StratifiedKFold

    y = np.array([s["label"] for s in samples])
    signers = np.array([s["signer"] for s in samples])
    uniq = sorted(set(signers))
    results = {}

    for name, make in _models(seed).items():
        res = {}
        # LOSO
        if len(uniq) >= 2:
            pred = np.empty(len(y), dtype=object)
            for held in uniq:
                te = signers == held
                tr = ~te
                if len(set(y[tr])) < 2:
                    pred[te] = REJ
                    continue
                m = make().fit(X[tr], y[tr])
                pred[te] = _predict_with_reject(m, X[te], threshold)
            res["loso"] = list(pred)
        # stratified k-fold (pooled)
        min_count = min(Counter(y).values())
        k = min(5, min_count)
        if k >= 2:
            pred = np.empty(len(y), dtype=object)
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            for tr, te in skf.split(X, y):
                m = make().fit(X[tr], y[tr])
                pred[te] = _predict_with_reject(m, X[te], threshold)
            res["kfold"] = list(pred)
            res["k"] = k
        results[name] = res
    return results


# ---------------------------------------------------------------- report

def _pct(d):
    if d["n"] == 0:
        return "n/a"
    lo, hi = d["ci95"]
    return f"{100*d['rate']:.1f}% ({d['k']}/{d['n']}; 95% CI {100*lo:.0f}–{100*hi:.0f}%)"


def _md_confusion(m, rows, cols) -> str:
    short = {REJ: "UNREC", NEG: "NEG"}
    head = "| true \\ pred | " + " | ".join(short.get(c, c) for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    body = [
        f"| {short.get(r, r)} | " + " | ".join(str(v) if v else "·" for v in m[i]) + " |"
        for i, r in enumerate(rows)
    ]
    return "\n".join([head, sep, *body])


def _plot_confusion(m, rows, cols, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _breakdown(samples, preds, key):
    groups = defaultdict(lambda: [0, 0])
    for s, p in zip(samples, preds):
        if s["label"] == NEG:
            continue
        g = groups[s[key]]
        g[1] += 1
        g[0] += p == s["label"]
    return {k: {"k": v[0], "n": v[1]} for k, v in sorted(groups.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="JSONL files exported from the Dataset recorder")
    ap.add_argument("--out", default=None, help="output folder (default: ../runs/eval/<timestamp>)")
    ap.add_argument("--reject-threshold", type=float, default=0.5,
                    help="learned models output UNRECOGNIZED below this probability (default 0.5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-learned", action="store_true", help="only evaluate the rule-based recognizer")
    ap.add_argument("--title", default="ISL Bridge evaluation")
    args = ap.parse_args(argv)

    catalog = load_catalog()
    sign_ids = [s["id"] for s in catalog["signs"]]
    samples = load_samples(args.files)
    unknown = sorted({s["label"] for s in samples} - set(sign_ids) - {NEG})
    if unknown:
        sys.exit(f"Unknown labels in data: {unknown}")
    if not samples:
        sys.exit("No samples found.")

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[2] / "runs" / "eval" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)

    y_true = [s["label"] for s in samples]
    signers = sorted({s["signer"] for s in samples})
    label_counts = Counter(y_true)
    rows = sign_ids + [NEG]
    cols = sign_ids + [REJ]

    metrics = {
        "title": args.title,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "files": [Path(f).name for f in args.files],
        "n_samples": len(samples),
        "signers": signers,
        "label_counts": dict(label_counts),
        "lighting_counts": dict(Counter(s["lighting"] for s in samples)),
    }

    # Rule-based
    rb_pred, rb_codes = run_rule_based(samples)
    metrics["rule_based"] = {
        **summarize(y_true, rb_pred, sign_ids),
        "unrecognized_codes": dict(rb_codes),
        "by_signer": _breakdown(samples, rb_pred, "signer"),
        "by_lighting": _breakdown(samples, rb_pred, "lighting"),
    }
    rb_cm = confusion(y_true, rb_pred, rows, cols)
    _plot_confusion(rb_cm, rows, cols, "Rule-based (deployed) recognizer", out / "confusion_rule_based.png")

    # Learned
    learned = {}
    if not args.no_learned:
        try:
            from evaluation.features import clip_features
            X = np.stack([clip_features(s["frames"]) for s in samples])
            raw = run_learned(samples, X, args.reject_threshold, args.seed)
            for name, res in raw.items():
                learned[name] = {}
                for proto in ("loso", "kfold"):
                    if proto in res:
                        pred = res[proto]
                        learned[name][proto] = summarize(y_true, pred, sign_ids)
                        if proto == "loso":
                            learned[name][proto]["by_signer"] = _breakdown(samples, pred, "signer")
                        cm = confusion(y_true, pred, rows, cols)
                        learned[name][proto]["_cm"] = cm
                if "k" in res:
                    learned[name]["k"] = res["k"]
        except ImportError as e:
            print(f"[warn] learned baselines skipped ({e}). pip install -r requirements-eval.txt")

    for name, res in learned.items():
        for proto in ("loso", "kfold"):
            if proto in res:
                _plot_confusion(res[proto]["_cm"], rows, cols, f"{name} ({proto})",
                                out / f"confusion_{name}_{proto}.png")

    metrics["learned"] = {
        n: {p: {k: v for k, v in r.items() if k != "_cm"} if isinstance(r, dict) else r for p, r in res.items()}
        for n, res in learned.items()
    }
    metrics["reject_threshold"] = args.reject_threshold
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    # Markdown report
    L = [f"# {args.title}", "",
         f"Generated {metrics['generated']} from {', '.join(metrics['files'])}.", "",
         f"**Data:** {len(samples)} clips (2.5 s of MediaPipe Hands landmarks each), "
         f"{len(signers)} signer(s): {', '.join(signers)}.", "",
         "| Label | Clips |", "|---|---|",
         *[f"| {l} | {label_counts.get(l, 0)} |" for l in rows], ""]
    if len(signers) < 2:
        L += ["> **Only one signer.** All numbers describe performance on that person's own hand. "
              "Record at least 3 signers to measure generalization (leave-one-signer-out).", ""]

    def block(name, m):
        return [f"### {name}", "",
                "| Metric | Value |", "|---|---|",
                f"| Sign accuracy | {_pct(m['sign_accuracy'])} |",
                f"| Wrong-sign rate | {_pct(m['wrong_sign_rate'])} |",
                f"| Rejection (UNRECOGNIZED) rate on signs | {_pct(m['rejection_rate'])} |",
                f"| False-accept rate on NEGATIVE clips | {_pct(m['false_accept_rate'])} |",
                f"| Macro-F1 (10 signs + NEGATIVE) | {m['macro_f1']:.3f} |", ""]

    L += ["## 1. Rule-based recognizer (deployed)", ""]
    L += block("All clips", metrics["rule_based"])
    rb = metrics["rule_based"]
    L += ["Per-sign recall: " + ", ".join(
        f"{c} {100*v['recall']:.0f}%" for c, v in rb["per_class"].items()), ""]
    if rb["unrecognized_codes"]:
        L += ["UNRECOGNIZED reasons: " + ", ".join(f"{k} ×{v}" for k, v in rb["unrecognized_codes"].items()), ""]
    if len(rb["by_signer"]) > 1:
        L += ["Accuracy by signer: " + ", ".join(
            f"{k} {v['k']}/{v['n']}" for k, v in rb["by_signer"].items()), ""]
    if len(rb["by_lighting"]) > 1:
        L += ["Accuracy by lighting: " + ", ".join(
            f"{k} {v['k']}/{v['n']}" for k, v in rb["by_lighting"].items()), ""]
    L += ["Confusion matrix (counts):", "", _md_confusion(rb_cm, rows, cols), "",
          "![rule-based confusion](confusion_rule_based.png)", ""]

    if learned:
        L += ["## 2. Learned baselines on landmark features", "",
              f"Untuned scikit-learn defaults; NEGATIVE trained as a class; "
              f"probability < {args.reject_threshold} → UNRECOGNIZED.", ""]
        if any("loso" in r for r in learned.values()):
            L += ["### Leave-one-signer-out (new-user estimate)", "",
                  "| Model | Sign acc. | Wrong-sign | Rejected | False-accept | Macro-F1 |",
                  "|---|---|---|---|---|---|"]
            L += [f"| rule-based (no training) | {100*rb['sign_accuracy']['rate']:.1f}% | "
                  f"{100*rb['wrong_sign_rate']['rate']:.1f}% | {100*rb['rejection_rate']['rate']:.1f}% | "
                  f"{100*rb['false_accept_rate']['rate']:.1f}% | {rb['macro_f1']:.3f} |"]
            for n, r in learned.items():
                if "loso" in r:
                    m = r["loso"]
                    L.append(f"| {n} | {100*m['sign_accuracy']['rate']:.1f}% | {100*m['wrong_sign_rate']['rate']:.1f}% | "
                             f"{100*m['rejection_rate']['rate']:.1f}% | {100*m['false_accept_rate']['rate']:.1f}% | "
                             f"{m['macro_f1']:.3f} |")
            L.append("")
        if any("kfold" in r for r in learned.values()):
            k = next(r["k"] for r in learned.values() if "k" in r)
            L += [f"### Stratified {k}-fold, pooled signers (optimistic: same people in train and test)", "",
                  "| Model | Sign acc. | Wrong-sign | Rejected | False-accept | Macro-F1 |",
                  "|---|---|---|---|---|---|"]
            for n, r in learned.items():
                if "kfold" in r:
                    m = r["kfold"]
                    L.append(f"| {n} | {100*m['sign_accuracy']['rate']:.1f}% | {100*m['wrong_sign_rate']['rate']:.1f}% | "
                             f"{100*m['rejection_rate']['rate']:.1f}% | {100*m['false_accept_rate']['rate']:.1f}% | "
                             f"{m['macro_f1']:.3f} |")
            L.append("")

    L += ["## Notes", "",
          "* Rule-based scores are rule weights, not calibrated probabilities.",
          "* Small per-class counts give wide confidence intervals; read differences between methods with care.",
          ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out / 'report.md'}")
    return metrics


if __name__ == "__main__":
    main()
