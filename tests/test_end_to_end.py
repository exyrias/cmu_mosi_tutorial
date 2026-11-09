import csv
import json
import sys

from mulg.cli import main


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["mulg", *map(str, args)])
    main()


def test_all_seven_training_checkpoint_reload_and_resume(tmp_path, monkeypatch):
    data = tmp_path / "synthetic.h5"
    output = tmp_path / "runs"
    invoke(monkeypatch, "smoke-data", "--output", data)
    args = [
        "run",
        "--data",
        data,
        "--output",
        output,
        "--epochs",
        "1",
        "--seeds",
        "13",
        "--hidden",
        "8",
        "--heads",
        "2",
        "--attention-layers",
        "1",
        "--batch-size",
        "8",
        "--micro-batch-size",
        "3",
        "--threads",
        "1",
        "--device",
        "cpu",
    ]
    invoke(monkeypatch, *args)
    results = json.loads((output / "results.json").read_text())
    assert {r["modalities"] for r in results} == {"t", "a", "v", "ta", "tv", "av", "tav"}
    assert all(r["synthetic"] and r["n_test"] == 7 and r["n_binary_nonzero"] == 6 for r in results)
    with open(output / "summary.csv") as stream:
        assert len(list(csv.DictReader(stream))) == 7
    evaluation = tmp_path / "evaluation"
    invoke(
        monkeypatch,
        "evaluate",
        "--checkpoint",
        output / "tav/seed_13/best.pt",
        "--data",
        data,
        "--output",
        evaluation,
        "--device",
        "cpu",
        "--threads",
        "1",
    )
    reloaded = json.loads((evaluation / "metrics.json").read_text())
    original = next(r for r in results if r["modalities"] == "tav")
    assert reloaded["Acc2"] == original["Acc2"]
    assert reloaded["Acc7"] == original["Acc7"]
    invoke(monkeypatch, *args, "--resume")
    assert json.loads((output / "results.json").read_text()) == results
