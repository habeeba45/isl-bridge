import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate import NEG, REJ, load_samples, main, summarize, wilson  # noqa: E402
from evaluation.synthetic import generate  # noqa: E402


def test_summarize_counts_error_types():
    signs = ["HELLO", "STOP"]
    y_true = ["HELLO", "HELLO", "HELLO", "STOP", NEG, NEG]
    y_pred = ["HELLO", "STOP", REJ, "STOP", REJ, "HELLO"]
    m = summarize(y_true, y_pred, signs)
    assert m["sign_accuracy"]["k"] == 2 and m["sign_accuracy"]["n"] == 4
    assert m["wrong_sign_rate"]["k"] == 1
    assert m["rejection_rate"]["k"] == 1
    assert m["false_accept_rate"]["k"] == 1 and m["false_accept_rate"]["n"] == 2


def test_wilson_bounds():
    lo, hi = wilson(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0


def test_load_skips_duplicate_ids(tmp_path):
    s = {"id": "a", "label": "STOP", "frames": []}
    f = tmp_path / "x.jsonl"
    f.write_text(json.dumps(s) + "\n" + json.dumps(s) + "\n", encoding="utf-8")
    assert len(load_samples([str(f)])) == 1


def test_end_to_end_on_synthetic(tmp_path):
    pytest.importorskip("sklearn")
    data = tmp_path / "syn.jsonl"
    with data.open("w", encoding="utf-8") as fh:
        for s in generate(n_signers=2, per_label=5, seed=1):
            fh.write(json.dumps(s) + "\n")
    out = tmp_path / "out"
    metrics = main([str(data), "--out", str(out), "--title", "SYNTHETIC test"])
    assert (out / "report.md").exists()
    assert (out / "metrics.json").exists()
    assert metrics["n_samples"] == 2 * 11 * 5
    assert "loso" in metrics["learned"]["logreg"]
