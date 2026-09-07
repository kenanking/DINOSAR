"""Evaluate a fine-tuned DINOSAR dual encoder on SARVLM."""

import argparse
import json
from pathlib import Path

import torch

from openclip_retrieval.config import (
    build_training_config,
    load_config_values,
)
from openclip_retrieval.data import (
    CsvImageTextDataset,
    build_eval_loader,
    build_eval_transform,
    build_tokenizer,
    read_csv_pairs,
)
from openclip_retrieval.evaluation import evaluate_retrieval_model
from openclip_retrieval.metrics import format_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/DINOSAR.py"))
    parser.add_argument("--csv", default="5k", help="'5k', 'full', or a CSV path")
    parser.add_argument("--image-root", type=Path, default=None, help="image root for a --csv path")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="metrics.jsonl to append to")
    return parser.parse_args()


def resolve_csv(csv_arg, image_root, data_cfg, config_path):
    if csv_arg == "5k":
        return data_cfg.eval_csv, data_cfg.eval_image_root
    if csv_arg == "full":
        return data_cfg.full_eval_csv, data_cfg.full_eval_image_root
    csv_path = Path(csv_arg)
    candidates = {
        data_cfg.eval_csv: data_cfg.eval_image_root,
        data_cfg.full_eval_csv: data_cfg.full_eval_image_root,
    }
    if image_root is not None:
        return csv_path, image_root
    for known_csv, root in candidates.items():
        if csv_path.resolve() == Path(known_csv).resolve():
            return csv_path, root
    raise SystemExit(f"{csv_arg} is not a configured CSV; pass --image-root (config: {config_path})")


def build_from_training_config(values, args):
    config = build_training_config(values)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = config.outputs.root / config.outputs.run_name / "last.pth"

    from openclip_retrieval.clip_model import build_clip_model, load_clip_checkpoint

    model = build_clip_model(config)
    load_clip_checkpoint(model, checkpoint)
    model.eval()
    tokenizer = build_tokenizer(config.text)
    transform = build_eval_transform(config.vision, config.transform)
    channels = config.vision.channels
    image_size = config.vision.image_size
    num_workers = config.data.num_workers if args.num_workers is None else args.num_workers
    prefetch_factor = config.data.prefetch_factor
    name = config.outputs.run_name
    data_cfg = config.data
    return (
        model, tokenizer, transform, channels, image_size,
        num_workers, prefetch_factor, name, data_cfg, checkpoint,
    )


def main():
    args = parse_args()
    values = load_config_values(args.config)
    built = build_from_training_config(values, args)
    (model, tokenizer, transform, channels, image_size, num_workers,
     prefetch_factor, name, data_cfg, checkpoint) = built

    csv_path, image_root = resolve_csv(args.csv, args.image_root, data_cfg, args.config)
    rows = read_csv_pairs(csv_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    dataset = CsvImageTextDataset(
        rows,
        image_root,
        transform,
        tokenizer,
        channels=channels,
        image_size=image_size,
        lenient=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = build_eval_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    metrics = evaluate_retrieval_model(model, loader, device)
    print(f"[{name}] csv={csv_path} n={len(rows)} checkpoint={checkpoint}")
    print(format_metrics(metrics))

    output = args.output
    if output is None:
        output = Path("outputs") / "eval" / name / "metrics.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "csv": str(csv_path),
        "rows": len(rows),
        "checkpoint": str(checkpoint),
        **metrics,
    }
    with open(output, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"appended: {output}")


if __name__ == "__main__":
    main()
