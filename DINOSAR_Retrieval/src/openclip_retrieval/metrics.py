"""Retrieval metrics, ported from the official SARVLM eval_retrieval.py.

Formulas (including torch.topk tie behavior and CPU computation) replicate
eval/RET/eval_retrieval.py exactly: R@K counts a hit when the ground-truth
diagonal index appears among the top-K columns; MeanRecall averages the six
R@{1,5,10} values of both directions.
"""

import torch


def recall_at_k(similarity_matrix: torch.Tensor, k: int, mode: str = "i2t") -> float:
    n = similarity_matrix.shape[0]

    if mode == "i2t":
        _, topk_indices = similarity_matrix.topk(k, dim=1)
        correct = torch.arange(n).unsqueeze(1).expand_as(topk_indices)
        hits = (topk_indices == correct).any(dim=1).float()
    else:
        sim_t2i = similarity_matrix.T
        _, topk_indices = sim_t2i.topk(k, dim=1)
        correct = torch.arange(n).unsqueeze(1).expand_as(topk_indices)
        hits = (topk_indices == correct).any(dim=1).float()

    recall = hits.mean().item() * 100.0
    return recall


def evaluate_retrieval(image_features: torch.Tensor, text_features: torch.Tensor) -> dict[str, float]:
    similarity_matrix = image_features @ text_features.T

    results = {}

    i2t_r1 = recall_at_k(similarity_matrix, 1, mode="i2t")
    i2t_r5 = recall_at_k(similarity_matrix, 5, mode="i2t")
    i2t_r10 = recall_at_k(similarity_matrix, 10, mode="i2t")
    i2t_mean = (i2t_r1 + i2t_r5 + i2t_r10) / 3.0

    results["I2T_R@1"] = i2t_r1
    results["I2T_R@5"] = i2t_r5
    results["I2T_R@10"] = i2t_r10
    results["I2T_MeanRecall"] = i2t_mean

    t2i_r1 = recall_at_k(similarity_matrix, 1, mode="t2i")
    t2i_r5 = recall_at_k(similarity_matrix, 5, mode="t2i")
    t2i_r10 = recall_at_k(similarity_matrix, 10, mode="t2i")
    t2i_mean = (t2i_r1 + t2i_r5 + t2i_r10) / 3.0

    results["T2I_R@1"] = t2i_r1
    results["T2I_R@5"] = t2i_r5
    results["T2I_R@10"] = t2i_r10
    results["T2I_MeanRecall"] = t2i_mean

    results["R@1"] = (i2t_r1 + t2i_r1) / 2.0
    results["R@5"] = (i2t_r5 + t2i_r5) / 2.0
    results["R@10"] = (i2t_r10 + t2i_r10) / 2.0
    results["MeanRecall"] = (i2t_mean + t2i_mean) / 2.0

    return results


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.2f}" for key, value in metrics.items())
