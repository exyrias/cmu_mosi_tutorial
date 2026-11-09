"""Independent training per modality set; checkpoint selection uses validation only."""

import csv
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import PreparedDataset, collate, sha256, write_json
from .metrics import evaluate
from .model import ModelConfig, MulG

MODALITIES = ("t", "a", "v", "ta", "tv", "av", "tav")


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def resolve_device(name):
    if name == "auto":
        name = (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return torch.device(name)


def loss_and_predictions(logits, labels, objective):
    if objective == "ce7":
        target = torch.round(labels.clamp(-3, 3)).long() + 3
        loss = F.cross_entropy(logits, target)
        score = (logits.softmax(-1) * torch.arange(-3, 4, device=logits.device)).sum(-1)
        classes = logits.argmax(-1)
    else:
        score = logits.squeeze(-1)
        loss = F.l1_loss(score, labels) if objective == "mae" else F.mse_loss(score, labels)
        classes = torch.round(score.clamp(-3, 3)).long() + 3
    return loss, score, classes


def minibatches(batch, size, device):
    features, masks, labels, ids = batch
    for start in range(0, len(labels), size):
        stop = min(start + size, len(labels))
        # Trim each microbatch to its own maximum valid length.
        x, m = {}, {}
        for modality in features:
            length = int(masks[modality][start:stop].sum(1).max())
            x[modality] = features[modality][start:stop, :length].to(device)
            m[modality] = masks[modality][start:stop, :length].to(device)
        yield x, m, labels[start:stop].to(device), ids[start:stop]


@torch.no_grad()
def predict(model, loader, objective, device, micro_batch):
    model.eval()
    targets, scores, classes, identifiers = [], [], [], []
    total_loss = total_count = 0
    for batch in loader:
        for x, masks, y, ids in minibatches(batch, micro_batch, device):
            loss, score, predicted = loss_and_predictions(model(x, masks), y, objective)
            total_loss += float(loss) * len(y)
            total_count += len(y)
            targets.extend(y.cpu().tolist())
            scores.extend(score.cpu().tolist())
            classes.extend(predicted.cpu().tolist())
            identifiers.extend(ids)
    if not total_count:
        raise ValueError("Cannot evaluate an empty split")
    return total_loss / total_count, targets, scores, classes, identifiers


def save_predictions(path, values):
    _, y, scores, classes, ids = values
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "target_score", "predicted_score", "predicted_class_minus3_to3"])
        writer.writerows(zip(ids, y, scores, np.asarray(classes) - 3))


def summarize(results, destination):
    write_json(destination / "results.json", results)
    rows = []
    for modality in dict.fromkeys(r["modalities"] for r in results):
        runs = [r for r in results if r["modalities"] == modality]
        row = {"modalities": modality, "runs": len(runs), "synthetic": runs[0]["synthetic"]}
        for metric in ["Acc2", "Acc2_with_zero", "Acc7", "MAE", "Corr"]:
            values = [r[metric] for r in runs if r[metric] is not None]
            row[metric + "_mean"] = float(np.mean(values)) if values else None
            row[metric + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
            )
        row["n_test"] = runs[0]["n_test"]
        row["n_binary_nonzero"] = runs[0]["n_binary_nonzero"]
        rows.append(row)
    if rows:
        with open(destination / "summary.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        "\n"
        + (
            "SYNTHETIC SMOKE TEST — NOT BENCHMARK RESULTS"
            if results and results[0]["synthetic"]
            else "CMU test-set results (%)"
        )
    )
    print(f"{'modalities':<12} {'Acc2 mean':>10} {'Acc7 mean':>10} {'runs':>5}")
    for row in rows:
        acc2 = row["Acc2_mean"]
        print(
            f"{row['modalities']:<12} {acc2 if acc2 is not None else float('nan'):>10.3f} {row['Acc7_mean']:>10.3f} {row['runs']:>5}"
        )


def source_hash():
    digest = hashlib.sha256()
    for name in ["train.py", "model.py", "data.py", "metrics.py"]:
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def run(args):
    if args.threads < 1 or args.micro_batch_size < 1 or args.patience < 0:
        raise ValueError("threads/micro-batch-size must be positive and patience nonnegative")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    with h5py.File(args.data, "r") as handle:
        manifest = json.loads(handle.attrs["manifest"])
    dataset = manifest["dataset"]
    epochs = args.epochs if args.epochs is not None else (100 if dataset == "mosi" else 50)
    batch_size = args.batch_size if args.batch_size is not None else (128 if dataset == "mosi" else 16)
    if epochs < 1 or batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs/batch-size/lr must be positive")
    modalities = list(MODALITIES) if args.modalities == ["all"] else args.modalities
    if len(set(modalities)) != len(modalities) or any(m not in MODALITIES for m in modalities):
        raise ValueError(f"Choose unique modality combinations from {MODALITIES}, or all")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be nonempty and unique")
    cfg = ModelConfig(
        hidden=args.hidden,
        heads=args.heads,
        attention_layers=args.attention_layers,
        gru_layers=args.gru_layers,
        text_dropout=args.text_dropout
        if args.text_dropout is not None
        else (0.2 if dataset == "mosi" else 0.3),
        attention_dropout=args.attention_dropout,
        output_dropout=args.output_dropout,
        gru_variant=args.gru_variant,
        objective=args.objective,
    )
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    data_hash = sha256(args.data)
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "device": str(device),
        "source_sha256": source_hash(),
    }
    setting = {
        "model": asdict(cfg),
        "epochs": epochs,
        "batch_size": batch_size,
        "micro_batch_size": args.micro_batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "clip_grad": args.clip_grad,
        "data_sha256": data_hash,
        "environment": environment,
    }
    signature = hashlib.sha256(json.dumps(setting, sort_keys=True).encode()).hexdigest()
    # Check all destinations before training so completed results cannot be overwritten silently.
    for modality in modalities:
        for seed in args.seeds:
            folder = destination / modality / f"seed_{seed}"
            if folder.exists() and any(folder.iterdir()):
                if not args.resume:
                    raise FileExistsError(f"Existing run {folder}; use --resume or a new --output")
                config_path = folder / "config.json"
                if not config_path.exists() or json.loads(config_path.read_text())["signature"] != signature:
                    raise ValueError(f"--resume configuration/data/source mismatch: {folder}")
    results = []
    for modality in modalities:
        for seed in args.seeds:
            folder = destination / modality / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / "metrics.json"
            if args.resume and result_path.exists():
                results.append(json.loads(result_path.read_text()))
                continue
            seed_everything(seed)
            sets = {s: PreparedDataset(args.data, s, modality) for s in ["train", "valid", "test"]}
            try:
                generator = torch.Generator().manual_seed(seed)
                loaders = {
                    s: DataLoader(
                        ds,
                        batch_size=batch_size,
                        shuffle=s == "train",
                        collate_fn=collate,
                        num_workers=0,
                        generator=generator if s == "train" else None,
                    )
                    for s, ds in sets.items()
                }
                model = MulG(manifest["input_dims"], modality, cfg).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                metadata = {
                    **setting,
                    "signature": signature,
                    "seed": seed,
                    "modalities": modality,
                    "input_dims": manifest["input_dims"],
                    "data_manifest": manifest,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "selection": "minimum validation loss; test is evaluated only after checkpoint selection",
                    "implementation": "documented reconstruction, not author code",
                }
                write_json(folder / "config.json", metadata)
                best_loss, best_epoch, stale = float("inf"), 0, 0
                started = time.monotonic()
                with open(folder / "history.csv", "w", newline="", encoding="utf-8") as stream:
                    history = csv.writer(stream)
                    history.writerow(["epoch", "train_loss", "valid_loss", "elapsed_seconds"])
                    for epoch in range(1, epochs + 1):
                        model.train()
                        total_loss = total_count = 0
                        for batch in loaders["train"]:
                            optimizer.zero_grad(set_to_none=True)
                            full_count = len(batch[2])
                            for x, masks, y, _ in minibatches(batch, args.micro_batch_size, device):
                                loss, _, _ = loss_and_predictions(model(x, masks), y, cfg.objective)
                                if not torch.isfinite(loss):
                                    raise FloatingPointError("Nonfinite training loss")
                                (loss * (len(y) / full_count)).backward()
                                total_loss += float(loss.detach()) * len(y)
                                total_count += len(y)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.clip_grad, error_if_nonfinite=True
                            )
                            optimizer.step()
                        valid = predict(model, loaders["valid"], cfg.objective, device, args.micro_batch_size)
                        if not np.isfinite(valid[0]):
                            raise FloatingPointError("Nonfinite validation loss")
                        history.writerow(
                            [epoch, total_loss / total_count, valid[0], time.monotonic() - started]
                        )
                        stream.flush()
                        print(
                            f"{dataset}/{modality}/seed={seed} epoch={epoch}/{epochs} train={total_loss / total_count:.5f} valid={valid[0]:.5f}",
                            flush=True,
                        )
                        if valid[0] < best_loss:
                            best_loss, best_epoch, stale = valid[0], epoch, 0
                            temporary = folder / "best.pt.part"
                            torch.save(
                                {
                                    "state_dict": model.state_dict(),
                                    "metadata": metadata,
                                    "best_epoch": best_epoch,
                                    "best_valid_loss": best_loss,
                                },
                                temporary,
                            )
                            temporary.replace(folder / "best.pt")
                        else:
                            stale += 1
                        if args.patience and stale >= args.patience:
                            break
                checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=True)
                model.load_state_dict(checkpoint["state_dict"])
                predictions = predict(model, loaders["test"], cfg.objective, device, args.micro_batch_size)
                save_predictions(folder / "test_predictions.csv", predictions)
                metrics = {
                    "dataset": dataset,
                    "synthetic": bool(manifest.get("synthetic", False)),
                    "modalities": modality,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "valid_loss": best_loss,
                    "test_loss": predictions[0],
                    "objective": cfg.objective,
                    "gru_variant": cfg.gru_variant,
                    **evaluate(predictions[1], predictions[2], predictions[3]),
                }
                write_json(result_path, metrics)
                results.append(metrics)
                summarize(results, destination)
            finally:
                for ds in sets.values():
                    ds.close()
    summarize(results, destination)


def evaluate_checkpoint(args):
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    metadata = checkpoint["metadata"]
    if sha256(args.data) != metadata["data_sha256"]:
        raise ValueError("Prepared data differs from the checkpoint's training data")
    dataset = PreparedDataset(args.data, "test", metadata["modalities"])
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        cfg = ModelConfig(**metadata["model"])
        model = MulG(metadata["input_dims"], metadata["modalities"], cfg).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        loader = DataLoader(dataset, batch_size=args.micro_batch_size, collate_fn=collate)
        predictions = predict(model, loader, cfg.objective, device, args.micro_batch_size)
        metrics = {
            "synthetic": bool(dataset.manifest.get("synthetic", False)),
            "modalities": metadata["modalities"],
            "seed": metadata["seed"],
            **evaluate(predictions[1], predictions[2], predictions[3]),
        }
        write_json(destination / "metrics.json", metrics)
        save_predictions(destination / "test_predictions.csv", predictions)
        print(json.dumps(metrics, indent=2))
    finally:
        dataset.close()
