import json

import h5py
import numpy as np

from mulg.data import PreparedDataset, prepare, sdk_manifest, temporal_indices


def test_interval_overlap_does_not_leak_boundary_frames():
    intervals = np.array([[0, 1], [1, 2], [2, 3], [1, 1], [2, 2]], dtype=float)
    np.testing.assert_array_equal(temporal_indices(intervals, 1, 2), [False, True, False, True, False])


def test_native_csd_official_splits_common_cohort_and_train_only_statistics(tmp_path):
    folds = sdk_manifest()["datasets"]["mosi"]["folds"]
    ids = {s: videos[0] for s, videos in folds.items()}
    paths = {m: tmp_path / f"{m}.csd" for m in ["t", "a", "v", "labels"]}
    for modality, path in paths.items():
        with h5py.File(path, "w") as handle:
            root = handle.create_group("root/data")
            for split, video in ids.items():
                group = root.create_group(video)
                if modality == "labels":
                    # Extra MOSEI emotion columns must never be used as sentiment targets.
                    group.create_dataset("features", data=[[1.2, 99], [-1.8, 99]])
                    group.create_dataset("intervals", data=[[0, 2], [3, 4]])
                else:
                    offset = 0 if split == "train" else 100
                    group.create_dataset("features", data=np.array([[0, 2], [2, 4]], dtype=float) + offset)
                    group.create_dataset("intervals", data=[[0, 1], [1, 2]])
    output = tmp_path / "prepared.h5"
    prepare("mosi", paths, output, caps=(128, 512, 512))
    with h5py.File(output) as handle:
        stats = json.loads(handle.attrs["stats"])
        manifest = json.loads(handle.attrs["manifest"])
    assert stats["a"]["mean"] == [1, 3]
    assert stats["a"]["std"] == [1, 1]
    assert manifest["counts"] == {"train": 1, "valid": 1, "test": 1}
    assert len(manifest["skipped"]) == 3
    assert manifest["synthetic"] is False
    ds = PreparedDataset(output, "test", "a")
    x, y, sample_id = ds[0]
    assert x["a"][0].tolist() == [99, 99]
    assert y == 1.2
    assert sample_id.startswith(ids["test"])
    ds.close()
