import socket
from multiprocessing import get_context

import pytest
import torch

from dinosar.losses import DINOLoss, sinkhorn_knopp


def _global_sinkhorn_teacher_probs(teacher_output, teacher_temp, n_iterations=3):
    q = torch.exp(teacher_output.float() / teacher_temp).t()
    batch_size = q.shape[1]
    num_prototypes = q.shape[0]

    q /= torch.sum(q)
    for _ in range(n_iterations):
        q /= torch.sum(q, dim=1, keepdim=True)
        q /= num_prototypes
        q /= torch.sum(q, dim=0, keepdim=True)
        q /= batch_size

    q *= batch_size
    return q.t()


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _distributed_teacher_probs_worker(
    rank,
    world_size,
    port,
    local_logits,
    teacher_temp,
    queue,
):
    import os

    import torch.distributed as dist

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        loss = DINOLoss(out_dim=local_logits.shape[-1], num_global_crops=2)
        local_probs = loss.build_teacher_probs(local_logits, teacher_temp=teacher_temp)
        queue.put((rank, local_probs.cpu()))
    finally:
        dist.destroy_process_group()


def test_dino_teacher_probs_are_row_normalized():
    loss = DINOLoss(out_dim=8, num_global_crops=2)
    teacher_logits = torch.randn(2, 3, 8)

    teacher_probs = loss.build_teacher_probs(teacher_logits, teacher_temp=loss.teacher_temp)

    assert teacher_probs.shape == teacher_logits.shape
    assert torch.allclose(teacher_probs.sum(dim=-1), torch.ones_like(teacher_probs[..., 0]), atol=1e-5, rtol=1e-5)


@pytest.mark.distributed
def test_dino_teacher_probs_match_global_sinkhorn_across_ranks():
    world_size = 2
    port = _find_free_port()
    teacher_temp = 0.07
    all_logits = torch.tensor(
        [
            [[0.2, -0.3, 1.1], [1.4, 0.5, -0.2]],
            [[-0.6, 0.7, 1.8], [0.9, -1.2, 0.1]],
            [[1.1, 0.4, -0.5], [0.3, 0.2, 1.6]],
            [[-0.7, 1.5, 0.6], [1.0, -0.1, 0.8]],
        ],
        dtype=torch.float32,
    )
    local_logits_by_rank = all_logits.chunk(world_size, dim=0)
    expected = _global_sinkhorn_teacher_probs(all_logits.flatten(0, 1), teacher_temp=teacher_temp).unflatten(
        0, all_logits.shape[:2]
    )

    ctx = get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_distributed_teacher_probs_worker,
            args=(rank, world_size, port, local_logits_by_rank[rank], teacher_temp, queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()

    results = {}
    for _ in range(world_size):
        rank, local_probs = queue.get(timeout=30)
        results[rank] = local_probs

    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    actual = torch.cat([results[rank] for rank in range(world_size)], dim=0)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Unified sinkhorn_knopp equivalence tests
# ---------------------------------------------------------------------------


def test_sinkhorn_knopp_matches_reference():
    """Unified function on flat (N, K) input must match the single-process reference.

    Uses moderate logit values where eps clamping in the unified function is
    a no-op, so the output is identical to the eps-free reference.
    """

    logits = torch.tensor(
        [
            [0.2, -0.3, 1.1, 1.4, 0.5, -0.2],
            [-0.6, 0.7, 1.8, 0.9, -1.2, 0.1],
            [1.1, 0.4, -0.5, 0.3, 0.2, 1.6],
            [-0.7, 1.5, 0.6, 1.0, -0.1, 0.8],
            [0.5, -0.8, 0.3, -0.4, 1.3, 0.2],
            [0.8, 0.1, -0.2, 0.6, 0.4, -0.3],
        ],
        dtype=torch.float32,
    )
    temp = 0.07

    expected = _global_sinkhorn_teacher_probs(logits, temp)
    actual = sinkhorn_knopp(logits, temp)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_sinkhorn_knopp_matches_dino_wrapper():
    """DINOLoss.build_teacher_probs (thin wrapper) must produce identical results."""

    torch.manual_seed(456)
    teacher_output = torch.randn(2, 4, 16)
    temp = 0.07

    loss = DINOLoss(out_dim=16, num_global_crops=2)
    wrapper_result = loss.build_teacher_probs(teacher_output, teacher_temp=temp)
    direct_result = sinkhorn_knopp(teacher_output.flatten(0, -2), temp).unflatten(0, teacher_output.shape[:-1])

    assert wrapper_result.shape == teacher_output.shape
    assert torch.allclose(wrapper_result, direct_result, atol=1e-7, rtol=1e-7)


def test_sinkhorn_knopp_ibot_shaped_input():
    """Variable-length masked-patch input: correct shape and row-normalized."""

    torch.manual_seed(789)
    n_masked = 37
    k = 64
    logits = torch.randn(n_masked, k)

    probs = sinkhorn_knopp(logits, teacher_temp=0.07)

    assert probs.shape == (n_masked, k)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)


def test_sinkhorn_knopp_eps_safety():
    """Near-zero and zero logits must not produce NaN or Inf."""

    logits_zero = torch.zeros(4, 8)
    probs_zero = sinkhorn_knopp(logits_zero, teacher_temp=0.07)
    assert torch.isfinite(probs_zero).all()

    logits_neg = torch.full((4, 8), -1e6)
    probs_neg = sinkhorn_knopp(logits_neg, teacher_temp=0.07)
    assert torch.isfinite(probs_neg).all()
