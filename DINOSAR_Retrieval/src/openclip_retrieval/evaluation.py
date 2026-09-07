"""Feature extraction and end-to-end retrieval evaluation.

Both the project ClipModel and native open_clip models expose
encode_image/encode_text with a normalize flag, so one evaluation path serves
trained DINOSAR dual-encoder evaluation. Features are
computed in eval precision (fp32) on the GPU and moved to CPU before the
similarity computation, matching the official eval script.
"""

import torch

from .metrics import evaluate_retrieval


@torch.no_grad()
def extract_features(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = model.training
    model.eval()
    image_features = []
    text_features = []
    for images, tokens, _image_ids in loader:
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        image_features.append(model.encode_image(images, normalize=True).float().cpu())
        text_features.append(model.encode_text(tokens, normalize=True).float().cpu())
    if was_training:
        model.train()
    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)


def evaluate_retrieval_model(model, loader, device) -> dict[str, float]:
    image_features, text_features = extract_features(model, loader, device)
    return evaluate_retrieval(image_features, text_features)
