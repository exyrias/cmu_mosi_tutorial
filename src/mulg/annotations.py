"""Verified MOSI sentiment annotations, distributed separately from Kaggle raw media."""

import csv
import math
import tempfile
from collections import Counter
from pathlib import Path

import requests

from .data import sdk_manifest, sha256, write_json

LABELS_URL = "https://drive.google.com/uc?export=download&id=1dqyQI8iOHGjwofgq1DvAD-inW-zrTa2h"
LABELS_REFERENCE = "https://github.com/thuiar/MMSA#2-datasets"
LABELS_SHA256 = "dec8b0affc5c7c923688040f091dddb61966afc729f54630ef91d3f51624b821"
LABELS_COUNT = 2199


def validate_labels(path):
    membership = {
        video: split
        for split, videos in sdk_manifest()["datasets"]["mosi"]["folds"].items()
        for video in videos
    }
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"video_id", "clip_id", "label", "mode"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected MMSA label.csv columns: video_id, clip_id, label, mode")
        rows = list(reader)
    seen, counts = set(), Counter()
    for row in rows:
        pair = (row["video_id"], row["clip_id"])
        if pair in seen:
            raise ValueError(f"Duplicate MOSI label: {pair}")
        seen.add(pair)
        value = float(row["label"])
        if not math.isfinite(value) or not -3 <= value <= 3:
            raise ValueError(f"Invalid continuous sentiment label for {pair}")
        if membership.get(row["video_id"]) != row["mode"]:
            raise ValueError(f"CSV split disagrees with official SDK video fold for {pair}")
        if not row["clip_id"].isdigit() or int(row["clip_id"]) < 1:
            raise ValueError(f"Invalid MOSI clip ID: {pair}")
        counts[row["mode"]] += 1
    if len(rows) != LABELS_COUNT:
        raise ValueError(f"Expected {LABELS_COUNT} MOSI annotations, found {len(rows)}")
    return dict(counts)


def download_labels(destination, overwrite=False):
    destination = Path(destination)
    if destination.exists() and not overwrite:
        if sha256(destination) != LABELS_SHA256:
            raise FileExistsError(f"Existing label CSV differs: {destination}; use another --output")
        counts = validate_labels(destination)
        print(f"Verified MOSI label CSV: {destination}; {counts}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mosi-labels-", dir=destination.parent) as temporary:
        target = Path(temporary) / "label.csv"
        try:
            # Kaggle credentials are deliberately not sent to this independent distribution host.
            with requests.get(LABELS_URL, stream=True, timeout=(20, 60)) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"MOSI annotation download returned HTTP {response.status_code}")
                with open(target, "wb") as stream:
                    stream.writelines(response.iter_content(1024 * 1024))
        except requests.RequestException as error:
            raise RuntimeError(f"MOSI annotation download failed ({type(error).__name__})") from None
        if sha256(target) != LABELS_SHA256:
            raise ValueError(
                "MOSI label checksum changed or a sign-in page was returned; inspect the source before using it"
            )
        counts = validate_labels(target)
        target.replace(destination)
    write_json(
        destination.with_suffix(".manifest.json"),
        {
            "dataset": "mosi",
            "source": "MMSA separate annotation distribution (not Kaggle)",
            "reference": LABELS_REFERENCE,
            "url": LABELS_URL,
            "sha256": LABELS_SHA256,
            "n_labels": LABELS_COUNT,
            "official_split_counts": counts,
            "target_column": "label (continuous score in [-3, 3]); not annotation/label_T/label_A/label_V",
        },
    )
    print(f"Downloaded MOSI annotations: {destination}; {counts}")
