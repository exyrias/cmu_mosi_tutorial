"""Extract explicitly documented features from Kaggle's pre-segmented MOSI files."""

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .annotations import validate_labels
from .data import cap_sequence, load_embeddings, sha256, write_json
from .kaggle import KAGGLE_PAGE, KAGGLE_VERSION


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def transcripts(raw_dir, rows):
    result = {}
    for video in sorted({row["video_id"] for row in rows}):
        path = raw_dir / "Raw/Transcript/Segmented" / f"{video}.annotprocessed"
        with open(path, encoding="utf-8-sig") as stream:
            for line in stream:
                number, separator, text = line.strip().partition("_DELIM_")
                if separator:
                    key = (video, str(int(number)))
                    if key in result:
                        raise ValueError(f"Duplicate transcript: {key}")
                    result[key] = text.strip()
    return result


class RawExtractors:
    def __init__(self, device="auto"):
        try:
            import cv2
            import opensmile
            import torchvision
            from torchvision.models.video import R3D_18_Weights, r3d_18
        except ImportError as error:
            raise RuntimeError(
                "Install raw feature dependencies: uv sync --frozen --extra cpu --extra raw --group dev"
            ) from error
        self.cv2 = cv2
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.IS09,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        weights = R3D_18_Weights.KINETICS400_V1
        self.transform = weights.transforms()
        self.video_model = r3d_18(weights=weights)
        self.video_model.fc = torch.nn.Identity()
        self.video_model.eval().to(self.device)
        self.spec = {
            "audio": "openSMILE IS09 low-level descriptors (not COVAREP MAT)",
            "visual": "torchvision R3D-18 KINETICS400_V1, frozen, fc removed, 512D",
            "visual_weights_url": weights.url,
            "visual_sampling": "approximately 8 fps; nonoverlapping 16-frame windows; repeat last frame for short final window",
            "opensmile_version": opensmile.__version__,
            "torchvision_version": torchvision.__version__,
            "opencv_version": cv2.__version__,
            "device": str(self.device),
        }

    def audio(self, path):
        return self.smile.process_file(str(path)).to_numpy(dtype=np.float32)

    @torch.inference_mode()
    def visual(self, path):
        capture = self.cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot decode video: {path}")
        outputs, window = [], []

        def encode():
            frames = window + [window[-1]] * (16 - len(window))
            tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            tensor = self.transform(tensor).unsqueeze(0).to(self.device)
            outputs.append(self.video_model(tensor).squeeze(0).cpu().numpy())

        try:
            fps = capture.get(self.cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid video frame rate: {path}")
            stride = max(1, round(fps / 8))
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    window.append(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
                    if len(window) == 16:
                        encode()
                        window.clear()
                index += 1
            if window:
                encode()
        finally:
            capture.release()
        if not outputs:
            raise ValueError(f"No video frames: {path}")
        return np.stack(outputs).astype(np.float32)


def prepare_kaggle(
    raw_dir, labels_csv, embedding_file, output, caps=(128, 512, 512), device="auto", threads=4
):
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    raw_dir, labels_csv, embedding_file, output = map(Path, (raw_dir, labels_csv, embedding_file, output))
    if output.exists():
        raise FileExistsError(f"Prepared data already exists: {output}; use another --output")
    validate_labels(labels_csv)
    with open(labels_csv, encoding="utf-8-sig", newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: (row["video_id"], int(row["clip_id"])))
    text = transcripts(raw_dir, rows)
    paths = {}
    for row in rows:
        pair = (row["video_id"], str(int(row["clip_id"])))
        if pair not in text or not words(text[pair]):
            raise ValueError(f"Missing/empty transcript for {pair}")
        stem = "_".join(pair)
        paths[pair] = {
            "a": raw_dir / "Raw/Audio/WAV_16000/Segmented" / f"{stem}.wav",
            "v": raw_dir / "Raw/Video/Segmented" / f"{stem}.mp4",
        }
        for path in paths[pair].values():
            if not path.is_file():
                raise FileNotFoundError(path)
    if not embedding_file.is_file():
        raise FileNotFoundError(f"Supply a pretrained plain-text GloVe/word2vec file: {embedding_file}")
    vocabulary = {word for value in text.values() for word in words(value)}
    embeddings, text_dim = load_embeddings(embedding_file, vocabulary)
    if not os.environ.get("TORCH_HOME"):
        torch.hub.set_dir(str(output.parent / "pretrained" / "torch" / "hub"))
    extractors = RawExtractors(device)
    audit = {
        "schema": 1,
        "dataset": "mosi",
        "synthetic": False,
        "source": "kaggle_raw",
        "url": KAGGLE_PAGE,
        "version": KAGGLE_VERSION,
        "labels": {"path": str(labels_csv), "sha256": sha256(labels_csv)},
        "embedding_file": {"path": str(embedding_file), "sha256": sha256(embedding_file)},
        "caps": dict(zip("tav", caps)),
        "features": extractors.spec,
        "text": "lowercase word tokens, fixed pretrained vectors; unknown words are zeros; no feature fitting on valid/test",
        "normalization": "train-only per-feature z-score on a/v after temporal pooling; text unchanged",
        "oov_tokens": 0,
        "total_tokens": 0,
        "nonfinite_values_replaced": dict.fromkeys("tav", 0),
        "inputs": [],
        "skipped": [],
    }
    download_manifest = raw_dir / "download_manifest.json"
    if download_manifest.is_file():
        audit["download_manifest_sha256"] = sha256(download_manifest)
    counts, dimensions, moments = Counter(), {}, {}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    try:
        with h5py.File(temporary, "w") as target:
            target.attrs["schema"] = 1
            for split in ["train", "valid", "test"]:
                target.create_group(split)
            for row in tqdm(rows, desc="Extract Kaggle MOSI (frozen encoders)"):
                pair = (row["video_id"], str(int(row["clip_id"])))
                tokens = words(text[pair])
                audit["total_tokens"] += len(tokens)
                audit["oov_tokens"] += sum(word not in embeddings for word in tokens)
                features = {
                    "t": np.stack(
                        [embeddings.get(word, np.zeros(text_dim, dtype=np.float32)) for word in tokens]
                    ),
                    "a": extractors.audio(paths[pair]["a"]),
                    "v": extractors.visual(paths[pair]["v"]),
                }
                split = row["mode"]  # validate_labels has checked every video against the official SDK folds.
                group = target[split].create_group(f"{counts[split]:08d}")
                group.attrs.update(id="_".join(pair), video_id=pair[0], y=float(row["label"]))
                for m, cap in zip("tav", caps):
                    x = np.asarray(features[m], dtype=np.float32)
                    if x.ndim != 2 or len(x) == 0 or x.shape[1] == 0:
                        raise ValueError(f"Empty/malformed {m} features for {pair}")
                    audit["nonfinite_values_replaced"][m] += int((~np.isfinite(x)).sum())
                    x = cap_sequence(np.nan_to_num(x, nan=0, posinf=0, neginf=0), cap)
                    if m in dimensions and dimensions[m] != x.shape[1]:
                        raise ValueError(f"Inconsistent {m} feature dimensions")
                    dimensions[m] = x.shape[1]
                    group.create_dataset(m, data=x, compression="gzip", compression_opts=1)
                    if split == "train" and m in "av":
                        n, total, square = moments.get(m, (0, np.zeros(x.shape[1]), np.zeros(x.shape[1])))
                        x64 = x.astype(np.float64)
                        moments[m] = (n + len(x), total + x64.sum(0), square + np.square(x64).sum(0))
                audit["inputs"].append(
                    {
                        "id": "_".join(pair),
                        "transcript_sha256": sha256(
                            raw_dir / "Raw/Transcript/Segmented" / f"{pair[0]}.annotprocessed"
                        ),
                        **{m + "_sha256": sha256(path) for m, path in paths[pair].items()},
                    }
                )
                counts[split] += 1
            if any(counts[split] == 0 for split in ["train", "valid", "test"]):
                raise ValueError("An official split is empty")
            stats = {}
            for m, (n, total, square) in moments.items():
                mean = total / n
                std = np.sqrt(np.maximum(square / n - mean**2, 0))
                std[std < 1e-6] = 1
                stats[m] = {"mean": mean.tolist(), "std": std.tolist()}
            audit.update(counts=dict(counts), input_dims=dimensions)
            target.attrs["manifest"] = json.dumps(audit, allow_nan=False)
            target.attrs["stats"] = json.dumps(stats, allow_nan=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(output.with_suffix(".manifest.json"), audit)
    print(f"Prepared Kaggle MOSI: {dict(counts)}; dimensions={dimensions}; {output}")
