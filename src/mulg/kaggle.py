"""Download the explicitly requested Kaggle MOSI release (raw media, not SDK CSDs)."""

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from tqdm import tqdm

from .data import sha256, write_json

KAGGLE_REF = "mathurinache/cmu-mosi"
KAGGLE_VERSION = 1
KAGGLE_PAGE = f"https://www.kaggle.com/datasets/{KAGGLE_REF}"
KAGGLE_DOWNLOAD = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_REF}"
RAW_REQUIREMENTS = (
    "The Kaggle release contains raw video/audio/transcripts and COVAREP MAT files. "
    "It does not contain sentiment ground truth or the four SDK CSD inputs. "
    "Downloading it alone does not enable Acc2/Acc7 evaluation. "
    "Use download-labels and prepare-kaggle with pretrained word vectors. See README.md."
)


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or ":" in name or not path.parts:
        raise ValueError(f"Unsafe dataset path: {name!r}")
    return Path(*path.parts)


def _credentials():
    # Never save credentials or signed redirect URLs in the manifest.
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return {"headers": {"Authorization": f"Bearer {token}"}}
    username, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if bool(username) != bool(key):
        raise ValueError("Set both KAGGLE_USERNAME and KAGGLE_KEY, or neither")
    return {"auth": (username, key)} if username else {}


def _download(url, target, description):
    digest = hashlib.sha256()
    try:
        with requests.get(
            url,
            params={"datasetVersionNumber": KAGGLE_VERSION},
            stream=True,
            timeout=(20, 120),
            **_credentials(),
        ) as response:
            if response.status_code != 200:
                message = f"Kaggle download returned HTTP {response.status_code}."
                if response.status_code in (401, 403):
                    message += (
                        " Configure local Kaggle credentials or download the ZIP in your browser "
                        "and pass --kaggle-archive. Do not paste credentials into chat."
                    )
                raise RuntimeError(message)
            total = int(response.headers.get("content-length", 0)) or None
            with (
                open(target, "wb") as stream,
                tqdm(total=total, unit="B", unit_scale=True, desc=description) as bar,
            ):
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
                    digest.update(block)
                    bar.update(len(block))
    except requests.RequestException as error:
        raise RuntimeError(f"Kaggle download failed ({type(error).__name__}). Retry the command.") from None
    return digest.hexdigest()


def _extract(archive, destination, expected_file=None):
    with zipfile.ZipFile(archive) as source:
        entries = [item for item in source.infolist() if not item.is_dir()]
        if not entries:
            raise ValueError("The Kaggle archive contains no files")
        # Validate every member before writing any of them.
        for entry in entries:
            safe_relative(entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Symlinks are not accepted in a dataset archive")
        if expected_file is not None and (
            len(entries) != 1 or PurePosixPath(entries[0].filename).name != expected_file.name
        ):
            raise ValueError("Single-file Kaggle response does not match the requested file")
        seen = set()
        for entry in entries:
            relative = expected_file if expected_file is not None else safe_relative(entry.filename)
            if relative in seen:
                raise ValueError(f"Duplicate archive path: {relative}")
            seen.add(relative)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as reader, open(target, "wb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _cached(destination, request, overwrite):
    if not destination.exists() or not any(destination.iterdir()):
        return False
    manifest_path = destination / "download_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"Refusing to replace unrelated files in {destination}; choose a new --output")
    manifest = json.loads(manifest_path.read_text())
    if any(manifest.get(key) != value for key, value in request.items()):
        raise ValueError(
            "Existing download has a different source/version/file selection; use another --output"
        )
    if overwrite:
        return False
    for record in manifest["files"]:
        path = destination / safe_relative(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(
                f"Cached download changed or is incomplete: {path}; use --overwrite to re-download"
            )
    print(f"Verified cached Kaggle data: {destination}\n{RAW_REQUIREMENTS}")
    return True


def download_kaggle(destination, filenames=None, archive=None, overwrite=False):
    destination = Path(destination).resolve()
    selected = sorted(set(filenames)) if filenames else None
    if selected:
        for name in selected:
            safe_relative(name)
            if not name.startswith("Raw/"):
                raise ValueError(
                    "This Kaggle release contains files under Raw/; specify the full Kaggle path"
                )
    if archive and selected:
        raise ValueError("--kaggle-file cannot be combined with --kaggle-archive")
    request = {
        "source": "kaggle",
        "dataset_ref": KAGGLE_REF,
        "version": KAGGLE_VERSION,
        "selected_files": selected,
    }
    if _cached(destination, request, overwrite):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mosi-kaggle-", dir=destination.parent) as temporary:
        scratch = Path(temporary)
        content = scratch / "content"
        content.mkdir()
        payload = scratch / "download.bin"
        archives = []
        if selected:
            for name in selected:
                digest = _download(f"{KAGGLE_DOWNLOAD}/{quote(name, safe='')}", payload, name)
                relative = safe_relative(name)
                if zipfile.is_zipfile(payload):
                    _extract(payload, content, expected_file=relative)
                else:
                    target = content / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    payload.replace(target)
                archives.append({"requested_file": name, "response_sha256": digest})
        else:
            if archive:
                payload = Path(archive).resolve()
                if not payload.is_file():
                    raise FileNotFoundError(payload)
                digest = sha256(payload)
            else:
                digest = _download(KAGGLE_DOWNLOAD, payload, "Kaggle MOSI v1")
            if not zipfile.is_zipfile(payload):
                raise ValueError("Expected a Kaggle dataset ZIP; the response may be an HTML sign-in page")
            _extract(payload, content)
            if not (content / "Raw").is_dir():
                raise ValueError("Expected the Raw/ directory from mathurinache/cmu-mosi version 1")
            archives.append({"requested_file": None, "response_sha256": digest})
        files = [
            {
                "path": str(path.relative_to(content).as_posix()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(content.rglob("*"))
            if path.is_file()
        ]
        manifest = {
            **request,
            "url": KAGGLE_PAGE,
            "layout": "raw_media",
            "training_ready": False,
            "missing_for_existing_csd_pipeline": [
                "sentiment_labels",
                "text_feature_csd",
                "audio_feature_csd",
                "visual_feature_csd",
            ],
            "note": RAW_REQUIREMENTS,
            "responses": archives,
            "files": files,
        }
        write_json(content / "download_manifest.json", manifest)
        # Replace only a directory owned by this downloader, after successful extraction.
        previous = scratch / "previous"
        if destination.exists():
            destination.replace(previous)
        try:
            content.replace(destination)
        except OSError:
            if previous.exists():
                previous.replace(destination)
            raise
    print(f"Downloaded Kaggle MOSI v1: {len(files)} files to {destination}\n{RAW_REQUIREMENTS}")
