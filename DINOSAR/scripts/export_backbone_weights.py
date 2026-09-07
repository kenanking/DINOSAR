#!/usr/bin/env python3
"""Export a backbone-only checkpoint for eval and downstream use."""

import argparse
from pathlib import Path

import torch

from dinosar.model import _load_checkpoint, extract_backbone_state_dict


def export_backbone_checkpoint(
    checkpoint_path,
    *,
    output_path,
):
    exported = extract_backbone_state_dict(_load_checkpoint(checkpoint_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output_path)
    return exported


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export a backbone-only checkpoint from DINOSAR training state.")
    parser.add_argument("--input", required=True, help="Path to the full DINOSAR training checkpoint.")
    parser.add_argument("--output", required=True, help="Path to the exported backbone checkpoint.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    export_backbone_checkpoint(args.input, output_path=args.output)


if __name__ == "__main__":
    main()
