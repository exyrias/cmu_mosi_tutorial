import io
import json
import pickle
import shutil
import sys

import h5py
import numpy as np
import pytest

from mulg import cli, drive
from mulg.data import PreparedDataset, collate, sdk_manifest, sha256
from mulg.model import ModelConfig, MulG
from mulg.train import MODALITIES


def fixture_release(tmp_path, monkeypatch, raw_ids=False):
    source = {}
    folds = sdk_manifest()["datasets"]["mosi"]["folds"]
    for split, ids in folds.items():
        offset = 0 if split == "train" else 100
        # Leading/trailing padding and an internal zero frame.
        text = np.array([[[0, 0], [1, 2], [0, 0], [3, 4], [0, 0]]], dtype=np.float32)
        audio = np.array([[[0], [2 + offset], [0 + offset], [4 + offset], [0]]], dtype=np.float64)
        vision = text.astype(np.float64)
        identifier = [f"{ids[0]}[0]"] if raw_ids else np.array([[f"{ids[0]}_1", "0.0", "1.0"]], dtype="S32")
        source[split] = {
            "text": text,
            "audio": audio,
            "vision": vision,
            "labels": np.array([[[0.2]]]),
            "id": identifier,
        }
    path = tmp_path / "mosi_data.pkl"
    path.write_bytes(pickle.dumps(source, protocol=2))
    info = {
        "id": "public-fixture",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "variant": "data",
        "counts": {s: 1 for s in folds},
        "input_dims": {"t": 2, "a": 1, "v": 2},
    }
    monkeypatch.setattr(drive, "DRIVE_FILES", {"mosi_data.pkl": info})
    return path, source


@pytest.mark.parametrize("raw_ids", [False, True])
def test_import_ids_padding_train_only_statistics_and_all_seven_models(tmp_path, monkeypatch, raw_ids):
    path, _ = fixture_release(tmp_path, monkeypatch, raw_ids)
    output = tmp_path / "prepared.h5"
    drive.prepare_drive(path, output)
    with h5py.File(output) as handle:
        stats = json.loads(handle.attrs["stats"])
        manifest = json.loads(handle.attrs["manifest"])
    assert stats["a"]["mean"] == [2.0]
    assert stats["a"]["std"][0] == pytest.approx(np.std([2.0, 0.0, 4.0]))
    assert manifest["counts"] == {"train": 1, "valid": 1, "test": 1}
    assert manifest["source"] == "google_drive"
    ds = PreparedDataset(output, "test")
    features, y, _ = ds[0]
    assert y == 0.2
    assert len(features["t"]) == 3
    assert features["t"][1].tolist() == [0, 0]  # Internal zero is not padding.
    assert features["a"][0, 0] > 50  # Test data did not affect the training statistics.
    x, masks, _, _ = collate([ds[0]])
    for modalities in MODALITIES:
        model = MulG(manifest["input_dims"], modalities, ModelConfig(hidden=8, heads=2, attention_layers=1))
        assert np.isfinite(model(x, masks).detach().numpy()).all()
    ds.close()


def test_reject_changed_checksum_without_creating_output(tmp_path, monkeypatch):
    path, _ = fixture_release(tmp_path, monkeypatch)
    path.write_bytes(path.read_bytes() + b"changed")
    output = tmp_path / "prepared.h5"
    with pytest.raises(ValueError, match="verified"):
        drive.prepare_drive(path, output)
    assert not output.exists()


def test_reject_wrong_official_split(tmp_path, monkeypatch):
    path, source = fixture_release(tmp_path, monkeypatch)
    source["test"]["id"] = source["train"]["id"].copy()
    path.write_bytes(pickle.dumps(source, protocol=2))
    drive.DRIVE_FILES["mosi_data.pkl"].update(bytes=path.stat().st_size, sha256=sha256(path))
    output = tmp_path / "prepared.h5"
    with pytest.raises(ValueError, match="Official SDK video fold"):
        drive.prepare_drive(path, output)
    assert not output.exists()
    assert not output.with_suffix(".h5.part").exists()


def test_restricted_reader_does_not_accept_other_constructors():
    class Unexpected:
        def __reduce__(self):
            return str, ("not a numeric array",)

    with pytest.raises(pickle.UnpicklingError, match="Unsupported object"):
        drive.NumericArrayReader(io.BytesIO(pickle.dumps(Unexpected()))).load()


def test_download_validates_cache_and_can_use_manual_file(tmp_path, monkeypatch):
    gdown = pytest.importorskip("gdown")
    source, _ = fixture_release(tmp_path, monkeypatch)
    calls = []

    def get(**kwargs):
        calls.append(kwargs)
        shutil.copyfile(source, kwargs["output"])
        return kwargs["output"]

    monkeypatch.setattr(gdown, "download", get)
    target = tmp_path / "downloaded"
    drive.download_drive(target)
    drive.download_drive(target)
    assert len(calls) == 1
    assert calls[0]["use_cookies"] is False
    assert drive.identify_release(target / "mosi_data.pkl")["variant"] == "data"
    assert json.loads((target / "mosi_data.pkl.manifest.json").read_text())["sha256"] == sha256(source)


def test_default_mosi_download_routes_to_drive(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "download_drive", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["mulg", "download", "--dataset", "mosi", "--output", str(tmp_path)])
    cli.main()
    assert calls == [(tmp_path, "mosi_data.pkl", False)]
