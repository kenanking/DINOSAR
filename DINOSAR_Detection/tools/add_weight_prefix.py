'''
Usage:
  # Convert a plain ViT/DINO-style checkpoint into this project's Detectron2 backbone format.
  python tools/add_weight_prefix.py weights/input.pth

  # Convert a nested state dict and save to an explicit output path.
  python tools/add_weight_prefix.py weights/input.pth --state-key model --out weights/input_detectron2.pth

  # Keep only backbone.* keys, strip that prefix, then add the Detectron2 ViTDet prefix.
  python tools/add_weight_prefix.py weights/input.pth \
    --keep-prefix backbone. \
    --strip-prefix backbone. \
    --prefix backbone.net.model.
'''
import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


DEFAULT_PREFIX = "backbone.net.model."
COMMON_STATE_KEYS = ("model", "state_dict", "teacher", "student")
KEYS_TO_REPORT = (
    "patch_embed.proj.weight",
    "cls_token",
    "mask_token",
    "storage_tokens",
    "local_cls_norm.weight",
    "local_cls_norm.bias",
    "norm.weight",
    "norm.bias",
)


def _shape(value: Any) -> str:
    return str(tuple(value.shape)) if hasattr(value, "shape") else type(value).__name__


def _extract_state_dict(checkpoint: Any, state_key: str | None) -> tuple[dict[str, Any], str | None]:
    if state_key:
        if not isinstance(checkpoint, dict) or state_key not in checkpoint:
            raise KeyError(f"State key '{state_key}' was not found in checkpoint")
        state_dict = checkpoint[state_key]
        return state_dict, state_key

    if isinstance(checkpoint, dict):
        for key in COMMON_STATE_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    return checkpoint, None


def add_prefix(state_dict: dict[str, Any], prefix: str) -> OrderedDict[str, Any]:
    prefixed = OrderedDict()
    for key, value in state_dict.items():
        prefixed[key if key.startswith(prefix) else f"{prefix}{key}"] = value
    return prefixed


def filter_keys(
    state_dict: dict[str, Any],
    *,
    keep_prefixes: tuple[str, ...],
    drop_prefixes: tuple[str, ...],
    drop_contains: tuple[str, ...],
    strip_prefixes: tuple[str, ...],
) -> OrderedDict[str, Any]:
    filtered = OrderedDict()
    for key, value in state_dict.items():
        if drop_prefixes and any(key.startswith(prefix) for prefix in drop_prefixes):
            continue
        if drop_contains and any(pattern in key for pattern in drop_contains):
            continue
        if keep_prefixes and not any(key.startswith(prefix) for prefix in keep_prefixes):
            continue

        new_key = key
        for prefix in strip_prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        filtered[new_key] = value
    return filtered


def expand_patch_embed_input_channels(
    state_dict: dict[str, Any],
    *,
    target_in_channels: int | None,
    key: str = "patch_embed.proj.weight",
) -> OrderedDict[str, Any]:
    if target_in_channels is None:
        return OrderedDict(state_dict)

    expanded = OrderedDict(state_dict)
    weight = expanded.get(key)
    if weight is None:
        raise KeyError(f"Patch embedding key '{key}' was not found")
    if not hasattr(weight, "shape") or len(weight.shape) != 4:
        raise ValueError(f"Patch embedding key '{key}' is not a 4D tensor")

    current_in_channels = weight.shape[1]
    if current_in_channels == target_in_channels:
        return expanded
    if current_in_channels != 1:
        raise ValueError(
            f"Can only expand single-channel patch embeddings; got {current_in_channels} channels"
        )

    expanded[key] = weight.repeat(1, target_in_channels, 1, 1) / target_in_channels
    return expanded


def print_summary(
    input_path: Path,
    output_path: Path,
    state_key: str | None,
    original: dict[str, Any],
    filtered: dict[str, Any],
    converted,
):
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Detected state key: {state_key or '<top-level>'}")
    print(f"Original keys: {len(original)}")
    print(f"Filtered keys: {len(filtered)}")
    print(f"Converted keys: {len(converted)}")
    print()
    print("Key shapes:")
    for key in KEYS_TO_REPORT:
        value = filtered.get(key)
        if value is None:
            value = original.get(key)
        print(f"  {key}: {_shape(value) if value is not None else '<missing>'}")
    print()
    print("Converted key samples:")
    for key in list(converted.keys())[:10]:
        print(f"  {key}: {_shape(converted[key])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a prefix to checkpoint state_dict keys.")
    parser.add_argument("input", type=Path, help="Input checkpoint path")
    parser.add_argument("--out", type=Path, default=None, help="Output checkpoint path")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Prefix to add, default: {DEFAULT_PREFIX}")
    parser.add_argument("--state-key", default=None, help="Explicit state_dict key, e.g. model or teacher")
    parser.add_argument(
        "--keep-prefix",
        action="append",
        default=[],
        help="Keep only keys starting with this prefix. Can be passed multiple times.",
    )
    parser.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="Drop keys starting with this prefix. Can be passed multiple times.",
    )
    parser.add_argument(
        "--drop-contains",
        action="append",
        default=[],
        help="Drop keys containing this text. Can be passed multiple times.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Strip this prefix from kept keys before adding the output prefix.",
    )
    parser.add_argument(
        "--expand-patch-embed-in-chans",
        type=int,
        default=None,
        help="Expand patch_embed.proj.weight from one input channel to this many channels.",
    )
    args = parser.parse_args()

    output_path = args.out or args.input.with_name(f"{args.input.stem}_detectron2{args.input.suffix}")
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict, detected_key = _extract_state_dict(checkpoint, args.state_key)
    filtered = filter_keys(
        state_dict,
        keep_prefixes=tuple(args.keep_prefix),
        drop_prefixes=tuple(args.drop_prefix),
        drop_contains=tuple(args.drop_contains),
        strip_prefixes=tuple(args.strip_prefix),
    )
    filtered = expand_patch_embed_input_channels(
        filtered,
        target_in_channels=args.expand_patch_embed_in_chans,
    )
    converted = add_prefix(filtered, args.prefix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": converted,
            "__author__": "add_weight_prefix.py",
            "matching_heuristics": False,
        },
        output_path,
    )
    print_summary(args.input, output_path, detected_key, state_dict, filtered, converted)


if __name__ == "__main__":
    main()
