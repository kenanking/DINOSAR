from pathlib import Path


image_size = 512
batch_size = 8
num_classes = 6
epochs = 12
milestone_epochs = [8, 11]
base_batch_size_for_lr = 16
warmup_steps = 500

CONFIGS_ROOT = Path(__file__).resolve().parent.parent


def checkpoint_output_dir(config_path: str, checkpoint: str) -> str:
    config_file = Path(config_path).resolve()
    checkpoint_stem = Path(checkpoint.split("?", 1)[0]).stem
    try:
        config_rel = config_file.relative_to(CONFIGS_ROOT).with_suffix("")
        return str(Path("output") / config_rel / checkpoint_stem)
    except ValueError:
        return str(Path("output") / config_file.stem / checkpoint_stem)
