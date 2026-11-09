"""Import checksum-pinned numeric MOSI releases from the user-selected Drive folder."""

import codecs
import json
import pickle
import re
import tempfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm import tqdm

from .data import cap_sequence, sdk_manifest, sha256, write_json

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq"
DRIVE_FILES = {
    "mosi_data.pkl": {
        "id": "1_XdzdW8UNG1TTS6QcX10uhoS6N11OBit",
        "bytes": 154041300,
        "sha256": "1a113ad5edc8b9b625a7e29e6b94a97ecade70494c4e022d9f7ba5affca3bbdc",
        "variant": "data",
        "counts": {"train": 1284, "valid": 229, "test": 686},
        "input_dims": {"t": 300, "a": 5, "v": 20},
    },
    "mosi_raw.pkl": {
        "id": "1szKIqO0t3Be_W91xvf6aYmsVVUa7wDHU",
        "bytes": 357186706,
        "sha256": "5b8a29b2a603ce37611117d088570369c35246f843dade8865ff59cc5ee6f338",
        "variant": "raw",
        "counts": {"train": 1283, "valid": 214, "test": 686},
        "input_dims": {"t": 300, "a": 74, "v": 35},
    },
}


def identify_release(path):
    path = Path(path)
    digest = sha256(path)
    for name, info in DRIVE_FILES.items():
        if digest == info["sha256"] and path.stat().st_size == info["bytes"]:
            return {"name": name, **info}
    raise ValueError(
        "Input does not match the verified mosi_data.pkl or mosi_raw.pkl release. "
        "Check that the download completed and is not a Drive HTML page. "
        "For this importer select a PKL file, not mosi.hdf5/mosi_unalign.hdf5."
    )


def download_drive(destination, filename="mosi_data.pkl", overwrite=False):
    if filename not in DRIVE_FILES:
        raise ValueError(f"Choose one of {list(DRIVE_FILES)}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    info = DRIVE_FILES[filename]
    if target.exists() and not overwrite:
        if identify_release(target)["name"] != filename:
            raise ValueError("Cached file does not match the selected Drive release")
        print(f"Verified cached Drive file: {target}")
    else:
        try:
            import gdown
            from gdown.exceptions import FileURLRetrievalError
        except ImportError as error:
            raise RuntimeError(
                "Install the downloader: uv sync --frozen --extra cpu --extra drive"
            ) from error
        with tempfile.TemporaryDirectory(prefix=".mosi-drive-", dir=destination) as temporary:
            path = Path(temporary) / filename
            try:
                result = gdown.download(id=info["id"], output=str(path), quiet=False, use_cookies=False)
            except (OSError, ValueError, requests.RequestException, FileURLRetrievalError) as error:
                raise RuntimeError(
                    f"Drive download failed ({type(error).__name__}). "
                    f"Download {filename} manually from {DRIVE_FOLDER}, then use prepare-drive --input."
                ) from None
            if result is None or not path.is_file():
                raise RuntimeError(
                    f"Download {filename} manually from {DRIVE_FOLDER} and use prepare-drive --input"
                )
            if identify_release(path)["name"] != filename:
                raise ValueError("Downloaded content does not match the selected Drive release")
            path.replace(target)
    write_json(
        destination / (filename + ".manifest.json"),
        {
            "source": "google_drive",
            "folder": DRIVE_FOLDER,
            "filename": filename,
            "file_id": info["id"],
            "sha256": info["sha256"],
            "bytes": info["bytes"],
        },
    )
    print(f"Drive input ready: {target}")


def _legacy_bytes(value, encoding, errors="strict"):
    if encoding.lower().replace("-", "") not in {"latin1", "utf8", "ascii"}:
        raise pickle.UnpicklingError("Unsupported legacy array encoding")
    return codecs.encode(value, encoding, errors)


class NumericArrayReader(pickle.Unpickler):
    """Accept only NumPy array reconstruction and the legacy byte codec."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("_codecs", "encode"): _legacy_bytes,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported object in dataset: {module}.{name}")
        return allowed[(module, name)]


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sample_id(row):
    value = row if isinstance(row, (str, bytes)) else row[0]
    value = _text(value)
    match = re.fullmatch(r"(.+)\[(\d+)\]", value)
    if match:
        return value, match[1]
    match = re.fullmatch(r"(.+)_(\d+)", value)
    if match:
        return value, match[1]
    raise ValueError(f"Unsupported MOSI segment ID: {value}")


def trim_shared_padding(features):
    """Trim only all-modality zero padding at the edges; preserve internal zeros."""
    lengths = {len(x) for x in features.values()}
    if len(lengths) != 1:
        raise ValueError("These verified PKL releases use aligned modality lengths")
    keep = np.logical_or.reduce([np.any(x != 0, axis=1) for x in features.values()])
    positions = np.flatnonzero(keep)
    start, end = (int(positions[0]), int(positions[-1]) + 1) if len(positions) else (0, 1)
    return {m: x[start:end] for m, x in features.items()}


def prepare_drive(path, output, caps=(128, 512, 512)):
    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError(f"Prepared data already exists: {output}; choose another --output")
    release = identify_release(path)  # Verify the entire file before interpreting pickle objects.
    with path.open("rb") as stream:
        source = NumericArrayReader(stream, encoding="latin1").load()
    if not isinstance(source, dict) or set(source) != {"train", "valid", "test"}:
        raise ValueError("Expected train/valid/test dictionaries in the Drive release")
    membership = {
        video: split
        for split, videos in sdk_manifest()["datasets"]["mosi"]["folds"].items()
        for video in videos
    }
    audit = {
        "schema": 1,
        "dataset": "mosi",
        "synthetic": False,
        "source": "google_drive",
        "folder": DRIVE_FOLDER,
        "release": release,
        "input_path": str(path),
        "input_dims": release["input_dims"],
        "caps": dict(zip("tav", caps)),
        "alignment": "provided aligned arrays, at most 50 steps; not raw asynchronous streams",
        "features": "numeric text/audio/vision as supplied; feature extraction details/selected dimensions are not specified in the file",
        "padding": "shared all-modality zero prefix/suffix trimmed; internal zero frames preserved",
        "normalization": "train-only per-feature z-score on a/v after padding removal and temporal pooling; text unchanged",
        "nonfinite_values_replaced": dict.fromkeys("tav", 0),
        "all_zero_sequences": dict.fromkeys("tav", 0),
        "skipped": [],
    }
    counts, moments, seen = Counter(), {}, set()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    try:
        with h5py.File(temporary, "w") as target:
            target.attrs["schema"] = 1
            for split in ["train", "valid", "test"]:
                data = source[split]
                if not isinstance(data, dict) or not {"text", "audio", "vision", "labels", "id"}.issubset(
                    data
                ):
                    raise ValueError(f"Missing features/labels/IDs in {split}")
                n = len(data["id"])
                if n != release["counts"][split]:
                    raise ValueError(f"Unexpected {split} sample count: {n}")
                labels = np.asarray(data["labels"])
                if labels.size != n or labels.dtype.kind not in "fi":
                    raise ValueError("Expected one numeric sentiment score per sample")
                labels = labels.reshape(-1)
                if not np.isfinite(labels).all() or np.any(np.abs(labels) > 3):
                    raise ValueError("Invalid sentiment labels")
                for m, key in [("t", "text"), ("a", "audio"), ("v", "vision")]:
                    x = data[key]
                    if not isinstance(x, np.ndarray) or x.dtype.kind not in "fi" or x.ndim != 3:
                        raise ValueError(f"Expected numeric NxTxD {key} array")
                    if x.shape[0] != n or x.shape[1] == 0 or x.shape[2] != release["input_dims"][m]:
                        raise ValueError(f"Unexpected {key} shape: {x.shape}")
                root = target.create_group(split)
                for i in tqdm(range(n), desc=f"Prepare Drive {split}"):
                    identifier, video = _sample_id(data["id"][i])
                    if membership.get(video) != split:
                        raise ValueError(f"Official SDK video fold disagrees for {identifier}")
                    if identifier in seen:
                        raise ValueError(f"Duplicate sample ID: {identifier}")
                    seen.add(identifier)
                    features = {}
                    for m, key in [("t", "text"), ("a", "audio"), ("v", "vision")]:
                        x = np.asarray(data[key][i], dtype=np.float32)
                        audit["nonfinite_values_replaced"][m] += int((~np.isfinite(x)).sum())
                        x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
                        audit["all_zero_sequences"][m] += int(not x.any())
                        features[m] = x
                    features = trim_shared_padding(features)
                    group = root.create_group(f"{i:08d}")
                    group.attrs.update(id=identifier, video_id=video, y=float(labels[i]))
                    for m, cap in zip("tav", caps):
                        x = cap_sequence(features[m], cap)
                        group.create_dataset(m, data=x, compression="gzip", compression_opts=1)
                        if split == "train" and m in "av":
                            n_frames, total, square = moments.get(
                                m, (0, np.zeros(x.shape[1]), np.zeros(x.shape[1]))
                            )
                            x64 = x.astype(np.float64)
                            moments[m] = (
                                n_frames + len(x),
                                total + x64.sum(0),
                                square + np.square(x64).sum(0),
                            )
                    counts[split] += 1
            stats = {}
            for m, (n_frames, total, square) in moments.items():
                mean = total / n_frames
                std = np.sqrt(np.maximum(square / n_frames - mean**2, 0))
                std[std < 1e-6] = 1
                stats[m] = {"mean": mean.tolist(), "std": std.tolist()}
            audit["counts"] = dict(counts)
            target.attrs["manifest"] = json.dumps(audit, allow_nan=False)
            target.attrs["stats"] = json.dumps(stats, allow_nan=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(output.with_suffix(".manifest.json"), audit)
    print(f"Prepared Drive {release['name']}: {dict(counts)}; dimensions={release['input_dims']}; {output}")
