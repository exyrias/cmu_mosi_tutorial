"""Read the SDK's native CSD files without its legacy alignment/global imports."""

import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from importlib.resources import files
from pathlib import Path

import h5py
import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def sdk_manifest():
    return json.loads(files("mulg").joinpath("resources/sdk_manifest.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def csd_group(handle, key=None):
    if key:
        if key not in handle or "data" not in handle[key]:
            raise ValueError(f"CSD root {key!r} with a data group is missing")
        return handle[key]["data"]
    roots = [name for name in handle if isinstance(handle[name], h5py.Group) and "data" in handle[name]]
    if len(roots) != 1:
        raise ValueError(f"Specify --*-key for a multi-root/nonstandard CSD; found roots: {roots}")
    return handle[roots[0]]["data"]


def download(
    dataset,
    destination,
    audio="COVAREP",
    visual="OpenFace_2",
    text="glove_vectors",
    scheme="https",
    overwrite=False,
):
    manifest = sdk_manifest()
    recipe = manifest["datasets"][dataset]
    selected = {
        "t": (recipe["raw"] if text == "words" else recipe["highlevel"])[text],
        "a": recipe["highlevel"][audio],
        "v": recipe["highlevel"][visual],
        "labels": next(iter(recipe["labels"].values())),
    }
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records, failures = {}, []
    for modality, original in selected.items():
        url = scheme + ":" + original.split(":", 1)[1]
        path = destination / (modality + ".csd")
        try:
            if not path.exists() or overwrite:
                temporary = path.with_suffix(".csd.part")
                try:
                    with requests.get(url, stream=True, timeout=(15, 60)) as response:
                        response.raise_for_status()
                        with (
                            open(temporary, "wb") as stream,
                            tqdm(
                                total=int(response.headers.get("content-length", 0)) or None,
                                unit="B",
                                unit_scale=True,
                                desc=modality,
                            ) as bar,
                        ):
                            for block in response.iter_content(1024 * 1024):
                                stream.write(block)
                                bar.update(len(block))
                    with h5py.File(temporary, "r") as handle:
                        csd_group(handle)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            with h5py.File(path, "r") as handle:
                csd_group(handle)
            records[modality] = {
                "url": url,
                "original_sdk_url": original,
                "path": str(path),
                "sha256": sha256(path),
            }
        except (requests.RequestException, OSError, ValueError) as error:
            failures.append(f"{modality}: {error}")
            records[modality] = {"url": url, "error": str(error)}
    write_json(
        destination / "download_manifest.json", {"sdk_commit": manifest["sdk_commit"], "files": records}
    )
    if failures:
        raise RuntimeError(
            "Download failed; no replacement/synthetic data were substituted.\n"
            + "\n".join(failures)
            + "\nThe SDK contains legacy URLs. Supply existing CSD files to prepare --text-csd / --audio-csd / --visual-csd / --labels-csd. See README.md."
        )


def read_sequence(group):
    x = np.asarray(group["features"])
    intervals = np.asarray(group["intervals"], dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or intervals.shape != (len(x), 2):
        raise ValueError("CSD features must be NxD and intervals Nx2")
    if not np.isfinite(intervals).all() or (intervals[:, 1] < intervals[:, 0]).any():
        raise ValueError("Invalid CSD intervals")
    order = np.argsort(intervals[:, 0], kind="stable")
    return x[order], intervals[order]


def temporal_indices(intervals, start, end):
    # Half-open overlap excludes frames that only touch a segment boundary.
    point = intervals[:, 0] == intervals[:, 1]
    return ((intervals[:, 1] > start) & (intervals[:, 0] < end)) | (
        point & (intervals[:, 0] >= start) & (intervals[:, 0] < end)
    )


def cap_sequence(x, cap):
    """Deterministic temporal mean pooling, preserving the entire segment."""
    if cap <= 0 or len(x) <= cap:
        return x
    return np.stack([chunk.mean(0) for chunk in np.array_split(x, cap)]).astype(np.float32)


def token_string(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def load_embeddings(path, vocabulary):
    result, dimension = {}, None
    with open(path, encoding="utf-8") as stream:
        for i, line in enumerate(tqdm(stream, desc="Reading word vectors")):
            pieces = line.rstrip().split()
            if i == 0 and len(pieces) == 2 and all(s.isdigit() for s in pieces):
                continue  # optional word2vec text header
            if len(pieces) < 3 or pieces[0] not in vocabulary:
                continue
            vector = np.asarray(pieces[1:], dtype=np.float32)
            dimension = len(vector) if dimension is None else dimension
            if len(vector) != dimension or not np.isfinite(vector).all():
                raise ValueError("Inconsistent/nonfinite embedding vectors")
            result[pieces[0]] = vector
    if not result:
        raise ValueError("No vocabulary items found in the plain-text embedding file")
    return result, dimension


def prepare(dataset, paths, output, caps=(128, 512, 512), keys=None, embedding_file=None):
    """Build label-segmented asynchronous sequences on one common t/a/v cohort."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            f"Prepared data already exists: {output}. Choose another --output or remove it explicitly."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = keys or {}
    manifest = sdk_manifest()
    folds = manifest["datasets"][dataset]["folds"]
    sets = {split: set(ids) for split, ids in folds.items()}
    if any(sets[a] & sets[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]):
        raise ValueError("Overlapping official video folds")
    membership = {video: split for split, ids in sets.items() for video in ids}
    audit = {
        "schema": 1,
        "dataset": dataset,
        "synthetic": False,
        "sdk_commit": manifest["sdk_commit"],
        "caps": dict(zip("tav", caps)),
        "normalization": "train-only per-feature z-score on a/v, after temporal pooling; text unchanged",
        "inputs": {m: {"path": str(p), "sha256": sha256(p)} for m, p in paths.items()},
        "counts": {},
        "skipped": [],
        "unknown_videos": [],
        "nonfinite_values_replaced": {m: 0 for m in "tav"},
    }
    if embedding_file:
        audit["embedding_file"] = {"path": str(embedding_file), "sha256": sha256(embedding_file)}
    counts, moments, dimensions = Counter(), {}, {}
    temporary = output.with_suffix(output.suffix + ".part")
    try:
        with ExitStack() as stack:
            groups = {
                m: csd_group(stack.enter_context(h5py.File(p, "r")), keys.get(m)) for m, p in paths.items()
            }
            embeddings, embedding_dim = None, None
            if embedding_file:
                vocabulary = set()
                for video in groups["t"]:
                    # No fit on validation/test: reading a fixed pretrained dictionary is feature extraction.
                    x, _ = read_sequence(groups["t"][video])
                    vocabulary.update(token_string(w) for w in x.reshape(-1))
                embeddings, embedding_dim = load_embeddings(embedding_file, vocabulary)
                audit["embedding_vocabulary_matches"] = len(embeddings)
                audit["oov_tokens"] = 0
            target = stack.enter_context(h5py.File(temporary, "w"))
            target.attrs["schema"] = 1
            for split in folds:
                target.create_group(split)
            for video in tqdm(sorted(groups["labels"]), desc=f"Prepare {dataset}"):
                video_base = video.split("[", 1)[0]
                split = membership.get(video_base)
                if split is None:
                    audit["unknown_videos"].append(video)
                    continue
                labels, intervals = read_sequence(groups["labels"][video])
                available = {}
                for m in "tav":
                    if video in groups[m]:
                        x, times = read_sequence(groups[m][video])
                    elif video_base in groups[m]:
                        x, times = read_sequence(groups[m][video_base])
                    else:
                        continue
                    if m == "t" and embeddings is not None:
                        if x.shape[1] != 1:
                            raise ValueError("--embedding-file requires a single-column raw word CSD")
                        tokens = [token_string(w) for w in x[:, 0]]
                        audit["oov_tokens"] += sum(w not in embeddings for w in tokens)
                        x = (
                            np.stack(
                                [embeddings.get(w, np.zeros(embedding_dim, dtype=np.float32)) for w in tokens]
                            )
                            if tokens
                            else np.empty((0, embedding_dim), dtype=np.float32)
                        )
                    if not np.issubdtype(x.dtype, np.number):
                        raise ValueError(
                            "Text contains raw words. Supply --embedding-file with plain-text GloVe/word2vec vectors."
                        )
                    audit["nonfinite_values_replaced"][m] += int((~np.isfinite(x)).sum())
                    x = np.nan_to_num(x.astype(np.float32), nan=0, posinf=0, neginf=0)
                    available[m] = (x, times)
                for i, (start, end) in enumerate(intervals):
                    sample_id = f"{video}::label_{i}"
                    reason = None
                    y = float(labels[i, 0])  # MOSEI column 0 is sentiment, not emotion labels.
                    if not np.isfinite(y) or not -3 <= y <= 3 or end <= start:
                        reason = "invalid label/interval"
                    features = {}
                    for m, cap in zip("tav", caps):
                        if m not in available:
                            reason = f"missing {m} video"
                            break
                        x, times = available[m]
                        cut = x[temporal_indices(times, start, end)]
                        if len(cut) == 0:
                            reason = f"empty {m} segment"
                            break
                        features[m] = cap_sequence(cut, cap)
                    if reason:
                        audit["skipped"].append({"id": sample_id, "split": split, "reason": reason})
                        continue
                    group = target[split].create_group(f"{counts[split]:08d}")
                    group.attrs.update(id=sample_id, video_id=video_base, y=y)
                    for m, x in features.items():
                        if m in dimensions and dimensions[m] != x.shape[1]:
                            raise ValueError(f"Inconsistent {m} dimensions")
                        dimensions[m] = x.shape[1]
                        group.create_dataset(m, data=x, compression="gzip", compression_opts=1)
                        if split == "train" and m in "av":
                            n, total, square = moments.get(m, (0, np.zeros(x.shape[1]), np.zeros(x.shape[1])))
                            x64 = x.astype(np.float64)
                            moments[m] = (n + len(x), total + x64.sum(0), square + np.square(x64).sum(0))
                    counts[split] += 1
            if any(counts[s] == 0 for s in folds):
                raise ValueError(
                    f"An official split has no retained samples: {dict(counts)}. Check file keys/timebases."
                )
            stats = {}
            for m, (n, total, square) in moments.items():
                mean = total / n
                std = np.sqrt(np.maximum(square / n - mean**2, 0))
                std[std < 1e-6] = 1
                stats[m] = {"mean": mean.tolist(), "std": std.tolist()}
            audit["counts"] = dict(counts)
            audit["input_dims"] = dimensions
            target.attrs["manifest"] = json.dumps(audit, allow_nan=False)
            target.attrs["stats"] = json.dumps(stats, allow_nan=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(output.with_suffix(".manifest.json"), audit)
    print(
        f"Prepared {output}: {dict(counts)}; skipped={len(audit['skipped'])}; unknown videos={len(audit['unknown_videos'])}"
    )


class PreparedDataset(Dataset):
    def __init__(self, path, split, modalities="tav"):
        self.path, self.split, self.modalities = str(path), split, modalities
        with h5py.File(path, "r") as handle:
            self.keys = sorted(handle[split])
            self.manifest = json.loads(handle.attrs["manifest"])
            self.stats = json.loads(handle.attrs["stats"])
        self.handle = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        group = self.handle[self.split][self.keys[index]]
        features = {}
        for m in self.modalities:
            x = torch.from_numpy(group[m][:])
            if m in self.stats:
                stats = self.stats[m]
                x = (x - torch.tensor(stats["mean"], dtype=x.dtype)) / torch.tensor(
                    stats["std"], dtype=x.dtype
                )
            features[m] = x
        return features, float(group.attrs["y"]), str(group.attrs["id"])

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def collate(samples):
    features, labels, ids = zip(*samples)
    x, masks = {}, {}
    for m in features[0]:
        lengths = torch.tensor([len(item[m]) for item in features])
        x[m] = torch.nn.utils.rnn.pad_sequence([item[m] for item in features], batch_first=True)
        masks[m] = torch.arange(x[m].shape[1])[None, :] < lengths[:, None]
    return x, masks, torch.tensor(labels, dtype=torch.float32), list(ids)


def synthetic(path, dataset="mosi"):
    """A small, visibly flagged offline fixture, never a benchmark result."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    dims = {"t": 8, "a": 6, "v": 10}
    counts = {"train": 21, "valid": 7, "test": 7}
    manifest = {
        "schema": 1,
        "dataset": dataset,
        "synthetic": True,
        "input_dims": dims,
        "counts": counts,
        "warning": "SYNTHETIC SMOKE TEST ONLY; not CMU benchmark accuracy",
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["manifest"] = json.dumps(manifest)
        handle.attrs["stats"] = "{}"
        handle.attrs["schema"] = 1
        for split, count in counts.items():
            root = handle.create_group(split)
            for i in range(count):
                group = root.create_group(f"{i:08d}")
                group.attrs.update(
                    id=f"SYNTHETIC_{split}_{i}", video_id=f"SYNTHETIC_{split}", y=float(i % 7 - 3)
                )
                for m, d in dims.items():
                    group.create_dataset(
                        m, data=rng.normal(size=(int(rng.integers(2, 8)), d)).astype(np.float32)
                    )
    write_json(path.with_suffix(".manifest.json"), manifest)
