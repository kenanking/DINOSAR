"""Loss functions for DINOv3 pretraining."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Sinkhorn-Knopp
# ---------------------------------------------------------------------------


@torch.no_grad()
def sinkhorn_knopp(teacher_logits, teacher_temp, n_iterations=3):
    """Distributed Sinkhorn-Knopp normalization for teacher soft-assignments.

    Produces approximately doubly-stochastic soft assignments from raw
    teacher logits.  Row normalization (across prototypes) is global via
    ``all_reduce``; column normalization (across samples) is rank-local.

    Args:
        teacher_logits: ``(N_local, K)`` raw logits from a teacher head.
        teacher_temp: temperature scalar.
        n_iterations: alternating normalization iterations (default 3).

    Returns:
        ``(N_local, K)`` normalized teacher probability targets.
    """

    q = torch.exp(teacher_logits.float() / teacher_temp).t()
    distributed = dist.is_available() and dist.is_initialized()

    n_local = torch.tensor([q.shape[1]], dtype=q.dtype, device=q.device)
    if distributed:
        dist.all_reduce(n_local)
    batch_size = n_local
    num_prototypes = q.shape[0]
    eps = torch.finfo(q.dtype).eps

    sum_q = torch.sum(q)
    if distributed:
        dist.all_reduce(sum_q)
    q /= sum_q.clamp_min(eps)

    for _ in range(n_iterations):
        sum_of_rows = torch.sum(q, dim=1, keepdim=True)
        if distributed:
            dist.all_reduce(sum_of_rows)
        q /= sum_of_rows.clamp_min(eps)
        q /= num_prototypes
        q /= torch.sum(q, dim=0, keepdim=True).clamp_min(eps)
        q /= batch_size

    q *= batch_size
    return q.t()


# ---------------------------------------------------------------------------
# DINO Loss
# ---------------------------------------------------------------------------


class DINOLoss(nn.Module):
    """Image-level DINO self-distillation loss."""

    def __init__(
        self,
        out_dim,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        student_temp=0.1,
        num_global_crops=2,
        warmup_teacher_temp_steps=0,
        ignore_global_diagonal=True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.num_global_crops = num_global_crops
        self.warmup_teacher_temp_steps = warmup_teacher_temp_steps
        self.ignore_global_diagonal = ignore_global_diagonal

    @torch.no_grad()
    def build_teacher_probs(self, teacher_output, teacher_temp, n_iterations=3):
        """Build teacher targets with Sinkhorn-Knopp normalization."""

        original_shape = teacher_output.shape[:-1]
        probs = sinkhorn_knopp(teacher_output.flatten(0, -2), teacher_temp, n_iterations)
        return probs.unflatten(0, original_shape)

    def _pairwise_loss(
        self,
        student_logits,
        teacher_probs,
        *,
        ignore_diagonal=False,
    ):
        student_log_probs = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        loss = -torch.einsum("s b k, t b k -> s t", student_log_probs, teacher_probs.float())
        batch_size = student_logits.shape[1]
        if ignore_diagonal:
            diagonal_terms = min(loss.shape[0], loss.shape[1])
            loss = torch.diagonal_scatter(loss, loss.new_zeros(diagonal_terms))
            denominator = (loss.numel() - diagonal_terms) * batch_size
            return loss.sum() / max(denominator, 1)
        return loss.sum() / (loss.numel() * batch_size)

    def forward(
        self,
        student_output_global,
        teacher_probs_global,
        student_output_local=None,
    ):
        with torch.autocast(device_type=student_output_global.device.type, enabled=False):
            global_loss = self._pairwise_loss(
                student_output_global,
                teacher_probs_global,
                ignore_diagonal=self.ignore_global_diagonal,
            )
            total_loss = global_loss

            local_loss = torch.zeros((), device=student_output_global.device)
            if student_output_local is not None and student_output_local.numel() > 0:
                local_loss = self._pairwise_loss(student_output_local, teacher_probs_global, ignore_diagonal=False)
                dino_global_terms = (
                    self.num_global_crops * (self.num_global_crops - 1)
                    if self.ignore_global_diagonal
                    else self.num_global_crops**2
                )
                dino_local_terms = self.num_global_crops * student_output_local.shape[0]
                total_terms = dino_global_terms + dino_local_terms
                global_scale = dino_global_terms / total_terms
                local_scale = dino_local_terms / total_terms
                total_loss = (global_scale * global_loss) + (local_scale * local_loss)

        metrics = {
            "dino_global_crops_loss": float(global_loss.item()),
            "dino_local_crops_loss": float(local_loss.item()),
            "dino_total_loss": float(total_loss.item()),
        }
        return total_loss, metrics


# ---------------------------------------------------------------------------
# KoLeo Loss
# ---------------------------------------------------------------------------


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic regularizer."""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_nns_inner(self, x):
        """Find nearest neighbors for L2-normalized vectors using inner products."""

        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)
        _, indices = torch.max(dots, dim=1)
        return indices

    def forward(self, student_output, eps=1e-8):
        with torch.autocast(student_output.device.type, enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            indices = self.pairwise_nns_inner(student_output)
            distances = self.pdist(student_output, student_output[indices])
            return -torch.log(distances + eps).mean()


# ---------------------------------------------------------------------------
# iBOT Patch Loss
# ---------------------------------------------------------------------------


def _softmax_cross_entropy(targets, student_logits, temp):
    return torch.sum(targets.float() * F.log_softmax(student_logits.float() / temp, dim=-1), dim=-1)


class iBOTPatchLoss(nn.Module):
    """Masked patch cross-entropy used by iBOT."""

    def __init__(self, patch_out_dim, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp

    @torch.no_grad()
    def build_teacher_probs(self, teacher_output, teacher_temp, n_iterations=3):
        """Build teacher targets with Sinkhorn-Knopp normalization."""

        return sinkhorn_knopp(teacher_output, teacher_temp, n_iterations)

    def forward(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        loss = _softmax_cross_entropy(teacher_patch_tokens_masked, student_patch_tokens_masked, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]


# ---------------------------------------------------------------------------
# Hidden Layer Distillation Loss
# ---------------------------------------------------------------------------


class HiddenLayerDistillationLoss(nn.Module):
    """Hidden layer distillation over intermediate ViT patch tokens."""

    VALID_LOSS_TYPES = {"zscore_mse", "cosine", "smooth_l1"}
    VALID_MASK_POLICIES = {"all", "visible", "masked"}

    def __init__(
        self,
        *,
        loss_type="zscore_mse",
        mask_policy="visible",
        block_weights=None,
        eps=1e-6,
    ):
        super().__init__()
        if loss_type not in self.VALID_LOSS_TYPES:
            raise ValueError(f"unknown hidden distillation loss_type: {loss_type}")
        if mask_policy not in self.VALID_MASK_POLICIES:
            raise ValueError(f"unknown hidden distillation mask_policy: {mask_policy}")
        self.loss_type = loss_type
        self.mask_policy = mask_policy
        self.block_weights = None if block_weights is None else tuple(float(weight) for weight in block_weights)
        self.eps = eps

    def _zscore(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt()

    def _loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        student = student.float()
        teacher = teacher.detach().float()
        if self.loss_type == "zscore_mse":
            return F.mse_loss(self._zscore(student), self._zscore(teacher))
        if self.loss_type == "cosine":
            return 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(student, teacher)
        raise AssertionError(f"unhandled hidden distillation loss_type: {self.loss_type}")

    def _validate_stacks(self, student_hidden_states, teacher_hidden_states):
        if len(student_hidden_states) != len(teacher_hidden_states):
            raise ValueError(
                "student and teacher hidden layer counts differ: "
                f"{len(student_hidden_states)} vs {len(teacher_hidden_states)}"
            )
        if self.block_weights is not None and len(self.block_weights) != len(student_hidden_states):
            raise ValueError(
                "hidden_distill.block_weights must match target_blocks length: "
                f"{len(self.block_weights)} vs {len(student_hidden_states)}"
            )

    def _apply_mask_policy(self, tokens: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
        if masks is None or self.mask_policy == "all":
            return tokens
        if masks.ndim != 2:
            raise ValueError(f"hidden distillation masks must be [B, Npatch], got {tuple(masks.shape)}")
        if tokens.shape[0] != masks.shape[0]:
            raise ValueError(f"hidden distillation mask batch mismatch: {tokens.shape[0]} vs {masks.shape[0]}")
        if tokens.shape[1] != masks.shape[1]:
            raise ValueError(
                "hidden distillation mask token mismatch: "
                f"tokens have {tokens.shape[1]} patch positions but mask has {masks.shape[1]}"
            )
        keep = masks if self.mask_policy == "masked" else ~masks
        if not bool(keep.any()):
            return tokens.new_empty((0, tokens.shape[-1]))
        return tokens[keep]

    def _select_tokens(self, tokens: torch.Tensor, masks: torch.Tensor | None, num_extra_tokens: int) -> torch.Tensor:
        patch_tokens = tokens[:, num_extra_tokens:]
        return self._apply_mask_policy(patch_tokens, masks)

    def _patch_metric(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        masks: torch.Tensor | None,
        num_extra_tokens: int,
        policy: str,
    ) -> torch.Tensor | None:
        if masks is None:
            return None
        patch_student = student[:, num_extra_tokens:]
        patch_teacher = teacher[:, num_extra_tokens:]
        if patch_student.shape[1] != masks.shape[1]:
            return None
        keep = masks if policy == "masked" else ~masks
        if not bool(keep.any()):
            return None
        return self._loss(patch_student[keep].detach(), patch_teacher[keep].detach()).detach()

    def forward(
        self, student_hidden_states, teacher_hidden_states, *, masks=None, num_extra_tokens=0, target_blocks=None
    ):
        self._validate_stacks(student_hidden_states, teacher_hidden_states)
        device = student_hidden_states[0][0].device
        total_loss = torch.zeros((), device=device)
        total_weight = 0.0
        metrics: dict[str, torch.Tensor] = {}
        patch_losses: list[torch.Tensor] = []
        masked_patch_losses: list[torch.Tensor] = []
        unmasked_patch_losses: list[torch.Tensor] = []
        blocks = list(range(len(student_hidden_states))) if target_blocks is None else list(target_blocks)
        if len(blocks) != len(student_hidden_states):
            raise ValueError(
                f"target_blocks length must match hidden states: {len(blocks)} vs {len(student_hidden_states)}"
            )

        for layer_index, (s_layer_views, t_layer_views) in enumerate(zip(student_hidden_states, teacher_hidden_states)):
            if len(s_layer_views) != len(t_layer_views):
                raise ValueError(
                    f"student and teacher view counts differ at hidden layer {layer_index}: "
                    f"{len(s_layer_views)} vs {len(t_layer_views)}"
                )
            layer_weight = 1.0 if self.block_weights is None else self.block_weights[layer_index]
            layer_loss = torch.zeros((), device=device)
            layer_terms = 0
            for view_index, (s_view, t_view) in enumerate(zip(s_layer_views, t_layer_views)):
                if s_view.shape != t_view.shape:
                    raise ValueError(
                        f"hidden state shape mismatch at layer {layer_index}, view {view_index}: "
                        f"{tuple(s_view.shape)} vs {tuple(t_view.shape)}"
                    )
                s = self._select_tokens(s_view, masks, int(num_extra_tokens))
                t = self._select_tokens(t_view, masks, int(num_extra_tokens))
                if s.shape != t.shape:
                    raise ValueError(
                        f"selected hidden state shape mismatch at layer {layer_index}, view {view_index}: "
                        f"{tuple(s.shape)} vs {tuple(t.shape)}"
                    )
                if s.numel() == 0:
                    continue
                view_loss = self._loss(s, t)
                layer_loss = layer_loss + view_loss
                layer_terms += 1

                with torch.no_grad():
                    patch_losses.append(
                        self._loss(
                            s_view[:, int(num_extra_tokens) :].detach(), t_view[:, int(num_extra_tokens) :].detach()
                        ).detach()
                    )
                    masked_loss = self._patch_metric(s_view, t_view, masks, int(num_extra_tokens), "masked")
                    unmasked_loss = self._patch_metric(s_view, t_view, masks, int(num_extra_tokens), "visible")
                    if masked_loss is not None:
                        masked_patch_losses.append(masked_loss)
                    if unmasked_loss is not None:
                        unmasked_patch_losses.append(unmasked_loss)

            if layer_terms:
                layer_loss = layer_loss / layer_terms
                total_loss = total_loss + layer_weight * layer_loss
                total_weight += layer_weight
                metrics[f"hidden_distill_block{blocks[layer_index]}_loss"] = layer_loss.detach()

        loss = total_loss / total_weight if total_weight > 0 else total_loss
        metrics["hidden_distill_loss"] = loss.detach()
        if patch_losses:
            metrics["hidden_distill_patch_loss"] = torch.stack(patch_losses).mean()
        if masked_patch_losses:
            metrics["hidden_distill_masked_patch_loss"] = torch.stack(masked_patch_losses).mean()
        if unmasked_patch_losses:
            metrics["hidden_distill_unmasked_patch_loss"] = torch.stack(unmasked_patch_losses).mean()
        return loss, metrics
