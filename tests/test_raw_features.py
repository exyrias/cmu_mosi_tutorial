import csv
import hashlib
import json

import h5py
import numpy as np
import pytest
from test_kaggle import Response

from mulg import annotations, raw_features
from mulg.data import PreparedDataset, sdk_manifest


def fixture_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(annotations, "LABELS_COUNT", 3)
    folds = sdk_manifest()["datasets"]["mosi"]["folds"]
    rows = [
        {"video_id": ids[0], "clip_id": "1", "label": "0.2", "mode": split} for split, ids in folds.items()
    ]
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def test_download_annotations_validates_checksum_official_folds_and_credential_separation(
    tmp_path, monkeypatch
):
    path, _ = fixture_labels(tmp_path, monkeypatch)
    payload = path.read_bytes()
    monkeypatch.setattr(annotations, "LABELS_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv("KAGGLE_API_TOKEN", "never-send-to-google")
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response(payload)

    monkeypatch.setattr(annotations.requests, "get", get)
    output = tmp_path / "download.csv"
    annotations.download_labels(output)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["official_split_counts"] == {"train": 1, "valid": 1, "test": 1}
    assert "headers" not in calls[0][1] and "auth" not in calls[0][1]
    annotations.download_labels(output)
    assert len(calls) == 1


def test_annotations_reject_wrong_fold_and_unverified_download(tmp_path, monkeypatch):
    path, rows = fixture_labels(tmp_path, monkeypatch)
    assert annotations.validate_labels(path) == {"train": 1, "valid": 1, "test": 1}
    path.write_text(path.read_text().replace(rows[0]["video_id"], "unknown_video"))
    with pytest.raises(ValueError, match="official SDK"):
        annotations.validate_labels(path)
    monkeypatch.setattr(annotations.requests, "get", lambda *a, **k: Response(b"<html>sign in</html>"))
    output = tmp_path / "bad.csv"
    with pytest.raises(ValueError, match="checksum changed"):
        annotations.download_labels(output)
    assert not output.exists()


def test_raw_preprocessing_uses_actual_transcript_continuous_label_and_train_only_statistics(
    tmp_path, monkeypatch
):
    labels, rows = fixture_labels(tmp_path, monkeypatch)
    root = tmp_path / "raw"
    split_for_video = {row["video_id"]: row["mode"] for row in rows}
    for row in rows:
        vid = row["video_id"]
        transcript = root / "Raw/Transcript/Segmented" / f"{vid}.annotprocessed"
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text("1_DELIM_HELLO WORLD\n")
        for folder, suffix in [("Raw/Audio/WAV_16000/Segmented", ".wav"), ("Raw/Video/Segmented", ".mp4")]:
            path = root / folder / (vid + "_1" + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"unit-test media fixture")
    vectors = tmp_path / "vectors.txt"
    vectors.write_text("hello 1 2\nworld 3 4\n")

    class Extractors:
        def __init__(self, device):
            self.spec = {"test_fixture": True}

        def audio(self, path):
            vid = path.stem.rsplit("_", 1)[0]
            offset = 0 if split_for_video[vid] == "train" else 100
            return np.array([[0], [2]], dtype=np.float32) + offset

        def visual(self, path):
            return np.array([[1, 3], [3, 5]], dtype=np.float32)

    monkeypatch.setattr(raw_features, "RawExtractors", Extractors)
    output = tmp_path / "prepared.h5"
    raw_features.prepare_kaggle(root, labels, vectors, output)
    with h5py.File(output) as handle:
        stats = json.loads(handle.attrs["stats"])
        manifest = json.loads(handle.attrs["manifest"])
    assert stats["a"] == {"mean": [1], "std": [1]}
    assert manifest["counts"] == {"train": 1, "valid": 1, "test": 1}
    assert manifest["input_dims"] == {"t": 2, "a": 1, "v": 2}
    assert manifest["oov_tokens"] == 0 and manifest["total_tokens"] == 6
    ds = PreparedDataset(output, "test")
    x, y, _ = ds[0]
    assert y == 0.2
    assert x["a"][0].tolist() == [99]
    assert x["t"].tolist() == [[1, 2], [3, 4]]
    ds.close()
