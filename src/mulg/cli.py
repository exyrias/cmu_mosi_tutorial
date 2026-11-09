import argparse
from pathlib import Path

from .annotations import download_labels
from .data import download, prepare, sdk_manifest, synthetic
from .drive import DRIVE_FILES, download_drive, prepare_drive
from .kaggle import RAW_REQUIREMENTS, download_kaggle
from .raw_features import prepare_kaggle
from .train import evaluate_checkpoint, run


def main():
    parser = argparse.ArgumentParser(
        description="MulG reconstruction: CMU-MOSI/MOSEI, all seven t/a/v subsets"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="Download MOSI from Drive (default), or choose Kaggle/SDK")
    p.add_argument("--dataset", choices=["mosi", "mosei"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--source", choices=["drive", "kaggle", "sdk"], help="Default: drive for MOSI, sdk for MOSEI"
    )
    p.add_argument("--drive-file", choices=list(DRIVE_FILES), help="Drive file; default mosi_data.pkl")
    p.add_argument(
        "--kaggle-file", action="append", help="Optional full Kaggle file path; repeat to download a subset"
    )
    p.add_argument(
        "--kaggle-archive", type=Path, help="Use a browser-downloaded MOSI v1 ZIP without API authentication"
    )
    p.add_argument("--audio-feature")
    p.add_argument("--visual-feature")
    p.add_argument("--text-feature", choices=["glove_vectors", "words"])
    p.add_argument("--scheme", choices=["http", "https"])
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser(
        "prepare-drive", help="Import downloaded Drive PKL features and labels (no network required)"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-text", type=int, default=128)
    p.add_argument("--max-audio", type=int, default=512)
    p.add_argument("--max-visual", type=int, default=512)
    p = sub.add_parser(
        "prepare-kaggle", help="Extract word vectors, openSMILE and 3D-CNN features from Kaggle MOSI"
    )
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--embedding-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto", help="Device for frozen video feature extraction")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-text", type=int, default=128)
    p.add_argument("--max-audio", type=int, default=512)
    p.add_argument("--max-visual", type=int, default=512)
    p = sub.add_parser(
        "download-labels", help="Download separately distributed, checksum-pinned MOSI sentiment labels"
    )
    p.add_argument(
        "--output", type=Path, required=True, help="Output CSV file, e.g. data/annotations/mosi.csv"
    )
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser(
        "prepare", help="Segment CSD files using labels; retain asynchronous modality timesteps"
    )
    p.add_argument("--dataset", choices=["mosi", "mosei"], required=True)
    p.add_argument("--raw-dir", type=Path, help="Directory containing t.csd/a.csd/v.csd/labels.csd")
    p.add_argument("--text-csd", type=Path)
    p.add_argument("--audio-csd", type=Path)
    p.add_argument("--visual-csd", type=Path)
    p.add_argument("--labels-csd", type=Path)
    for name in ["text", "audio", "visual", "labels"]:
        p.add_argument(f"--{name}-key", help="Optional HDF5 root name")
    p.add_argument(
        "--embedding-file",
        type=Path,
        help="Plain-text word2vec/GloVe vectors when --text-csd contains raw words",
    )
    p.add_argument("--max-text", type=int, default=128)
    p.add_argument("--max-audio", type=int, default=512)
    p.add_argument("--max-visual", type=int, default=512)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run", help="Train each modality set independently and report test Acc2/Acc7")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modalities", nargs="+", default=["all"], help="all, or e.g. t a v ta tv av tav")
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73])
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, help="Effective batch size, defaults to paper Table 1")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--patience", type=int, default=0, help="0: no early stopping (default); >0: validation patience"
    )
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=40)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--attention-layers", type=int, default=2)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--text-dropout", type=float)
    p.add_argument("--attention-dropout", type=float, default=0.1)
    p.add_argument("--output-dropout", type=float, default=0.1)
    p.add_argument("--gru-variant", choices=["paper", "pytorch"], default="paper")
    p.add_argument("--objective", choices=["ce7", "mae", "mse"], default="ce7")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed identical runs; restart interrupted runs from epoch 1",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser(
        "evaluate", help="Re-evaluate one saved checkpoint on its original prepared test split"
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p = sub.add_parser("smoke-data", help="Create SYNTHETIC data to test installation; not benchmark data")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        source = args.source or ("drive" if args.dataset == "mosi" else "sdk")
        if source == "drive":
            if args.dataset != "mosi":
                parser.error("The specified Drive folder contains MOSI only")
            if any(
                [
                    args.kaggle_file,
                    args.kaggle_archive,
                    args.audio_feature,
                    args.visual_feature,
                    args.text_feature,
                    args.scheme,
                ]
            ):
                parser.error("Kaggle/SDK-specific options cannot be used with the Drive source")
            download_drive(args.output, args.drive_file or "mosi_data.pkl", args.overwrite)
            return
        if args.drive_file:
            parser.error("--drive-file requires the Drive source")
        if source == "kaggle":
            if args.dataset != "mosi":
                parser.error("The requested Kaggle source is CMU-MOSI only; use --source sdk for MOSEI")
            if any([args.audio_feature, args.visual_feature, args.text_feature, args.scheme]):
                parser.error("--*-feature and --scheme apply only to --source sdk")
            download_kaggle(args.output, args.kaggle_file, args.kaggle_archive, args.overwrite)
            return
        if args.kaggle_file or args.kaggle_archive:
            parser.error("--kaggle-file/--kaggle-archive require the Kaggle source")
        args.audio_feature = args.audio_feature or "COVAREP"
        args.visual_feature = args.visual_feature or "OpenFace_2"
        args.text_feature = args.text_feature or "glove_vectors"
        args.scheme = args.scheme or "https"
        recipe = sdk_manifest()["datasets"][args.dataset]
        if args.audio_feature not in recipe["highlevel"] or args.visual_feature not in recipe["highlevel"]:
            parser.error(f"Feature unavailable in SDK recipe; highlevel keys: {list(recipe['highlevel'])}")
        download(
            args.dataset,
            args.output,
            args.audio_feature,
            args.visual_feature,
            args.text_feature,
            args.scheme,
            args.overwrite,
        )
    elif args.command == "prepare-drive":
        if min(args.max_text, args.max_audio, args.max_visual) < 0:
            parser.error("Sequence caps must be >=0; 0 disables the cap")
        prepare_drive(args.input, args.output, (args.max_text, args.max_audio, args.max_visual))
    elif args.command == "download-labels":
        download_labels(args.output, args.overwrite)
    elif args.command == "prepare-kaggle":
        if min(args.max_text, args.max_audio, args.max_visual) < 0:
            parser.error("Sequence caps must be >=0; 0 disables the cap")
        prepare_kaggle(
            args.raw_dir,
            args.labels_csv,
            args.embedding_file,
            args.output,
            (args.max_text, args.max_audio, args.max_visual),
            args.device,
            args.threads,
        )
    elif args.command == "prepare":
        paths, keys = {}, {}
        for m, name in [("t", "text"), ("a", "audio"), ("v", "visual"), ("labels", "labels")]:
            path = getattr(args, name + "_csd")
            paths[m] = path if path else (args.raw_dir / (m + ".csd") if args.raw_dir else None)
            keys[m] = getattr(args, name + "_key")
        if any(p is None or not p.is_file() for p in paths.values()):
            if args.raw_dir and (args.raw_dir / "Raw").is_dir():
                parser.error(
                    RAW_REQUIREMENTS
                    + " prepare requires t.csd/a.csd/v.csd/labels.csd or explicit --*-csd paths."
                )
            parser.error("Supply four existing CSD files with --raw-dir and/or --*-csd")
        if min(args.max_text, args.max_audio, args.max_visual) < 0:
            parser.error("Sequence caps must be >=0; 0 disables the cap")
        prepare(
            args.dataset,
            paths,
            args.output,
            (args.max_text, args.max_audio, args.max_visual),
            keys,
            args.embedding_file,
        )
    elif args.command == "run":
        run(args)
    elif args.command == "evaluate":
        evaluate_checkpoint(args)
    else:
        synthetic(args.output)


if __name__ == "__main__":
    main()
