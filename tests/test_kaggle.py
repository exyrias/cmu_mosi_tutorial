import io
import json
import sys
import zipfile

import pytest

from mulg import cli, kaggle


def zip_bytes(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for path, content in members.items():
            archive.writestr(path, content)
    return stream.getvalue()


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.headers = {"content-length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, _):
        yield self.payload[:7]
        yield self.payload[7:]


def test_full_download_pins_version_hashes_and_reuses_verified_cache(tmp_path, monkeypatch):
    calls = []
    payload = zip_bytes({"Raw/Transcript/Segmented/video.annotprocessed": b"1_DELIM_text"})

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response(payload)

    monkeypatch.setattr(kaggle.requests, "get", get)
    out = tmp_path / "raw"
    kaggle.download_kaggle(out)
    manifest = json.loads((out / "download_manifest.json").read_text())
    assert manifest["source"] == "kaggle"
    assert manifest["dataset_ref"] == "mathurinache/cmu-mosi"
    assert manifest["version"] == 1
    assert manifest["training_ready"] is False
    assert manifest["files"][0]["sha256"] == kaggle.sha256(out / manifest["files"][0]["path"])
    assert calls[0][1]["params"] == {"datasetVersionNumber": 1}
    kaggle.download_kaggle(out)
    assert len(calls) == 1
    (out / manifest["files"][0]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="Cached download changed"):
        kaggle.download_kaggle(out)
    kaggle.download_kaggle(out, overwrite=True)
    assert len(calls) == 2


@pytest.mark.parametrize("wrapped", [False, True])
def test_single_file_download_handles_raw_or_server_zip_and_hides_credentials(tmp_path, monkeypatch, wrapped):
    name = "Raw/Audio/WAV_16000/COVAREP/video.mat"
    data = b"MATLAB data fixture"
    payload = zip_bytes({"video.mat": data}) if wrapped else data
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response(payload)

    monkeypatch.setattr(kaggle.requests, "get", get)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "private-test-value")
    out = tmp_path / "raw"
    kaggle.download_kaggle(out, filenames=[name])
    assert (out / name).read_bytes() == data
    assert "Raw%2FAudio%2F" in calls[0][0]
    assert calls[0][1]["headers"]["Authorization"] == "Bearer private-test-value"
    assert "private-test-value" not in (out / "download_manifest.json").read_text()
    with pytest.raises(ValueError, match="different source/version/file selection"):
        kaggle.download_kaggle(out)


def test_browser_archive_does_not_use_network(tmp_path, monkeypatch):
    archive = tmp_path / "browser.zip"
    archive.write_bytes(zip_bytes({"Raw/Transcript/Full/video.textonly": b"text"}))
    monkeypatch.setattr(kaggle.requests, "get", lambda *a, **k: pytest.fail("Unexpected network call"))
    out = tmp_path / "raw"
    kaggle.download_kaggle(out, archive=archive)
    assert (out / "Raw/Transcript/Full/video.textonly").read_bytes() == b"text"
    assert archive.exists()


def test_http_failure_leaves_no_completed_output(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle.requests, "get", lambda *a, **k: Response(b"forbidden", status=403))
    out = tmp_path / "raw"
    with pytest.raises(RuntimeError, match="HTTP 403"):
        kaggle.download_kaggle(out)
    assert not out.exists()
    assert not list(tmp_path.glob(".mosi-kaggle-*"))


def test_rejects_path_escape_and_unrelated_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(zip_bytes({"../escape": b"bad"}))
    with pytest.raises(ValueError, match="Unsafe dataset path"):
        kaggle.download_kaggle(tmp_path / "raw", archive=archive)
    assert not (tmp_path / "escape").exists()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="unrelated"):
        kaggle.download_kaggle(unrelated, archive=archive, overwrite=True)
    assert (unrelated / "keep.txt").read_text() == "keep"


def test_cli_routes_mosi_to_kaggle_and_keeps_explicit_sdk(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "download_kaggle", lambda *a: calls.append(("kaggle", a)))
    monkeypatch.setattr(cli, "download", lambda *a: calls.append(("sdk", a)))
    monkeypatch.setattr(sys, "argv", ["mulg", "download", "--dataset", "mosi", "--output", str(tmp_path)])
    cli.main()
    assert calls[-1][0] == "kaggle"
    monkeypatch.setattr(
        sys, "argv", ["mulg", "download", "--dataset", "mosi", "--source", "sdk", "--output", str(tmp_path)]
    )
    cli.main()
    assert calls[-1][0] == "sdk"
    assert calls[-1][1][2:6] == ("COVAREP", "OpenFace_2", "glove_vectors", "https")


def test_prepare_raw_kaggle_fails_before_creating_fake_training_data(tmp_path, monkeypatch, capsys):
    (tmp_path / "Raw").mkdir()
    output = tmp_path / "prepared.h5"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mulg", "prepare", "--dataset", "mosi", "--raw-dir", str(tmp_path), "--output", str(output)],
    )
    with pytest.raises(SystemExit):
        cli.main()
    assert "does not contain sentiment ground truth" in capsys.readouterr().err
    assert not output.exists()
