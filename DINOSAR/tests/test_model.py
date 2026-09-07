import pytest
import torch
import torch.nn.functional as F

from dinosar.losses import DINOLoss, HiddenLayerDistillationLoss, KoLeoLoss, iBOTPatchLoss
from dinosar.model import (
    DINO,
    StudentTeacherWrapper,
    _apply_backbone_weights,
    _extract_flat_state_dict,
    _load_checkpoint,
    extract_backbone_state_dict,
    load_backbone_from_config,
    vit_base,
)


def test_forward_features_list_collects_hidden_states():
    backbone = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1)
    x_list = [torch.randn(2, 1, 64, 64)]
    masks_list = [None]
    outputs, hidden_states = backbone.forward_features_list(x_list, masks_list, hidden_distill_blocks=(0, 3, 7, 11))
    assert len(outputs) == 1
    assert len(hidden_states) == 4
    assert all(len(layer_views) == 1 for layer_views in hidden_states)
    assert hidden_states[0][0].shape == (2, 17, 384)


def test_student_teacher_wrapper_returns_hidden_states():
    student = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1)
    model = StudentTeacherWrapper.from_backbones(student=student, out_dim=128)
    views = [torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)]
    outputs = model(views, hidden_distill_blocks=(0, 11))
    assert "hidden_states" in outputs["student_global"]
    assert "hidden_states" in outputs["teacher_global"]
    assert len(outputs["student_global"]["hidden_states"]) == 2
    assert outputs["student_global"]["hidden_states"][0][0].shape == (4, 17, student.embed_dim)
    assert outputs["student_global"]["hidden_num_extra_tokens"] == 1


def test_patch_only_hidden_distillation_excludes_register_tokens():
    student = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1, num_register_tokens=4)
    model = StudentTeacherWrapper.from_backbones(student=student, out_dim=128)
    views = [torch.randn(2, 1, 224, 224), torch.randn(2, 1, 224, 224)]
    outputs = model(views, hidden_distill_blocks=(11,))

    hidden = outputs["student_global"]["hidden_states"][0][0]
    num_extra_tokens = outputs["student_global"]["hidden_num_extra_tokens"]
    patch_hidden = hidden[:, num_extra_tokens:]
    loss_fn = HiddenLayerDistillationLoss(loss_type="smooth_l1")
    student_hidden = hidden.clone()
    teacher_hidden = hidden.clone()
    teacher_hidden[:, :num_extra_tokens] = teacher_hidden[:, :num_extra_tokens] + 100.0

    loss, metrics = loss_fn([[student_hidden]], [[teacher_hidden]], num_extra_tokens=num_extra_tokens)

    assert hidden.shape[1] == 1 + 4 + 196
    assert num_extra_tokens == 5
    assert patch_hidden.shape[1] == 196
    assert loss.item() == 0.0
    assert metrics["hidden_distill_patch_loss"].item() == 0.0


def test_student_teacher_wrapper_linear_hidden_predictor_is_trainable():
    student = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1)
    model = StudentTeacherWrapper.from_backbones(
        student=student,
        out_dim=128,
        hidden_distill_blocks=(0, 11),
        hidden_distill_use_predictor=True,
    )
    views = [torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)]
    outputs = model(views, hidden_distill_blocks=(0, 11))
    assert set(model.hidden_distill_predictors.keys()) == {"0", "11"}
    assert model.hidden_distill_predictors["0"].weight.requires_grad
    assert outputs["student_global"]["hidden_states"][0][0].shape == (4, 17, student.embed_dim)


def test_student_teacher_wrapper_no_hidden_states_by_default():
    student = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1)
    model = StudentTeacherWrapper.from_backbones(student=student, out_dim=128)
    views = [torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)]
    outputs = model(views)
    assert "hidden_states" not in outputs["student_global"]


def test_hidden_layer_distillation_loss_finite():
    loss_fn = HiddenLayerDistillationLoss()
    student_hidden = [[torch.randn(2, 16, 384)] for _ in range(4)]
    teacher_hidden = [[torch.randn(2, 16, 384)] for _ in range(4)]
    loss, metrics = loss_fn(student_hidden, teacher_hidden)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "hidden_distill_loss" in metrics


def test_hidden_layer_distillation_zero_for_identical():
    loss_fn = HiddenLayerDistillationLoss()
    x = torch.randn(2, 8, 64)
    loss, _ = loss_fn([[x]], [[x.clone()]])
    assert loss.item() < 1e-5


def test_hidden_layer_distillation_excludes_masked_patches():
    loss_fn = HiddenLayerDistillationLoss(loss_type="smooth_l1", mask_policy="visible")
    student_hidden = torch.zeros(1, 5, 2)
    teacher_hidden = student_hidden.clone()
    teacher_hidden[:, 1] = 10.0
    teacher_hidden[:, 4] = 10.0
    masks = torch.tensor([[True, False, False, True]])
    loss, metrics = loss_fn([[student_hidden]], [[teacher_hidden]], masks=masks, num_extra_tokens=1)
    assert loss.item() == 0.0
    assert metrics["hidden_distill_masked_patch_loss"].item() > 0.0
    assert metrics["hidden_distill_unmasked_patch_loss"].item() == 0.0


def test_hidden_layer_distillation_always_ignores_extra_tokens():
    loss_fn = HiddenLayerDistillationLoss(loss_type="smooth_l1")
    student_hidden = torch.zeros(1, 5, 2)
    teacher_hidden = student_hidden.clone()
    teacher_hidden[:, :1] = 10.0
    loss, _ = loss_fn([[student_hidden]], [[teacher_hidden]], num_extra_tokens=1)
    assert loss.item() == 0.0


def test_hidden_layer_distillation_rejects_shape_mismatch():
    loss_fn = HiddenLayerDistillationLoss()
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_fn([[torch.randn(1, 4, 8)]], [[torch.randn(1, 5, 8)]])


def test_student_teacher_wrapper_forward_shapes():
    student = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1)
    teacher = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1)
    model = StudentTeacherWrapper.from_backbones(student=student, teacher=teacher, out_dim=1024)
    views = [torch.randn(2, 1, 224, 224), torch.randn(2, 1, 224, 224)]
    views.extend(torch.randn(2, 1, 96, 96) for _ in range(4))
    outputs = model(views)
    assert outputs["student_global"]["cls_after_head"].shape == (2, 2, 1024)
    assert outputs["student_global"]["cls_pre_head"].shape == (2, 2, model.student.embed_dim)
    assert outputs["teacher_global"]["cls_after_head"].shape == (2, 2, 1024)
    assert outputs["student_local"]["cls_after_head"].shape == (4, 2, 1024)


def test_vit_base_forward_features():
    model = vit_base(
        patch_size=16,
        in_chans=1,
        num_register_tokens=2,
        layerscale_init=1.0e-5,
        mask_k_bias=True,
    )
    x = torch.randn(2, 1, 224, 224)
    outputs = model.forward_features(x)
    assert outputs["x_norm_clstoken"].shape == (2, 768)
    assert outputs["x_norm_regtokens"].shape == (2, 2, 768)
    assert outputs["x_norm_patchtokens"].shape == (2, 196, 768)


def test_dino_forward_projects_cls_tokens():
    model = DINO(backbone=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1), out_dim=256)
    x = torch.randn(2, 1, 224, 224)
    outputs = model(x)
    assert outputs.shape == (2, 256)


def test_forward_features_list_reuses_rope_per_view(monkeypatch):
    backbone = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1)
    x_list = [torch.randn(2, 1, 64, 64), torch.randn(2, 1, 32, 32)]
    masks_list = [None, None]

    call_count = {"value": 0}
    original_forward = backbone.rope_embed.forward

    def wrapped_forward(*args, **kwargs):
        call_count["value"] += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(backbone.rope_embed, "forward", wrapped_forward)

    backbone.forward_features_list(x_list, masks_list)

    assert call_count["value"] == len(x_list)


def test_backbone_checkpoint_loading(tmp_path):
    checkpoint_path = tmp_path / "backbone.pth"
    torch.save({"model": {"patch_embed.proj.weight": torch.randn(384, 1, 16, 16)}}, checkpoint_path)
    state_dict = _extract_flat_state_dict(_load_checkpoint(checkpoint_path))
    assert "patch_embed.proj.weight" in state_dict


def test_apply_backbone_weights_ignores_storage_tokens_for_zero_register_target():
    source = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1, num_register_tokens=4)
    target = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1, num_register_tokens=0)

    with torch.no_grad():
        source.cls_token.fill_(0.25)

    load_result, ignored_keys = _apply_backbone_weights(target, source.state_dict(), strict=True)

    assert ignored_keys == ["storage_tokens"]
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    assert torch.allclose(target.cls_token, source.cls_token)


def test_load_backbone_from_config_uses_model_config(tmp_path):
    checkpoint_path = tmp_path / "backbone.pth"
    source = DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1, num_register_tokens=4)
    with torch.no_grad():
        source.cls_token.fill_(0.25)
        source.storage_tokens.fill_(0.5)
    torch.save(source.state_dict(), checkpoint_path)

    loaded = load_backbone_from_config(
        {"arch": "vit_small", "patch_size": 16, "in_chans": 1, "num_register_tokens": 4},
        checkpoint_path,
    )

    assert loaded.storage_tokens is not None
    assert loaded.storage_tokens.shape == (1, 4, loaded.embed_dim)
    assert torch.allclose(loaded.cls_token, source.cls_token)
    assert torch.allclose(loaded.storage_tokens, source.storage_tokens)


def test_extract_backbone_state_dict_strips_orig_mod_prefix():
    checkpoint = {
        "model_state_dict": {
            "teacher.backbone._orig_mod.patch_embed.proj.weight": torch.randn(384, 1, 16, 16),
        }
    }
    state_dict = extract_backbone_state_dict(checkpoint)
    assert "patch_embed.proj.weight" in state_dict
    assert "_orig_mod.patch_embed.proj.weight" not in state_dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile requires CUDA")
def test_compile_forward_and_backward():
    torch.manual_seed(42)
    model = StudentTeacherWrapper.from_backbones(
        student=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        teacher=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        out_dim=128,
    ).cuda()
    model.train()
    views = [torch.randn(2, 1, 64, 64, device="cuda"), torch.randn(2, 1, 64, 64, device="cuda")]
    views.extend(torch.randn(2, 1, 32, 32, device="cuda") for _ in range(2))
    global_masks = torch.zeros(4, (64 // 16) ** 2, dtype=torch.bool, device="cuda")
    global_masks[:, ::3] = True
    masks = [global_masks, None]
    mask_indices_list = global_masks.flatten().nonzero().flatten()
    torch.manual_seed(0)
    eager_outputs = model(views, masks=masks, mask_indices_list=mask_indices_list)
    eager_loss = eager_outputs["student_global"]["cls_after_head"].sum()
    eager_loss.backward()
    eager_grad = model.student.backbone.cls_token.grad.clone()
    model.zero_grad()
    compiled_model = torch.compile(model, mode="default", dynamic=True)
    torch.manual_seed(0)
    compiled_outputs = compiled_model(views, masks=masks, mask_indices_list=mask_indices_list)
    compiled_loss = compiled_outputs["student_global"]["cls_after_head"].sum()
    compiled_loss.backward()
    compiled_grad = model.student.backbone.cls_token.grad.clone()
    assert torch.allclose(
        eager_outputs["student_global"]["cls_after_head"],
        compiled_outputs["student_global"]["cls_after_head"],
        atol=5e-3,
        rtol=1e-3,
    )
    assert torch.allclose(eager_grad, compiled_grad, atol=5e-3, rtol=1e-3)


def test_all_losses_return_finite_scalars():
    # DINO loss
    dino_loss_fn = DINOLoss(out_dim=256, num_global_crops=2, warmup_teacher_temp_steps=10)
    student_global = torch.randn(2, 4, 256)
    teacher_global = torch.randn(2, 4, 256)
    teacher_probs = dino_loss_fn.build_teacher_probs(teacher_global, teacher_temp=dino_loss_fn.teacher_temp)
    dino_loss, _ = dino_loss_fn(student_global, teacher_probs, student_output_local=None)
    assert torch.isfinite(dino_loss)

    # KoLeo loss
    koleo_input = torch.tensor([[0.2, 0.1, -0.7, 0.4], [0.1, -0.4, 0.3, 0.9], [-0.5, 0.6, 0.2, -0.1]])
    koleo_loss = KoLeoLoss()(koleo_input)
    assert koleo_loss.ndim == 0 and torch.isfinite(koleo_loss)

    # iBOT patch loss
    student_masks_flat = torch.tensor([[True, False, True], [False, True, True]])
    masks_weight = (
        (1 / student_masks_flat.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(student_masks_flat)[student_masks_flat]
    )
    student_patches = torch.randn(4, 4)
    teacher_patches = F.softmax(torch.randn(4, 4), dim=-1)
    ibot_loss = iBOTPatchLoss(4)(
        student_patches,
        teacher_patches,
        student_masks_flat=student_masks_flat,
        n_masked_patches=int(student_masks_flat.sum().item()),
        masks_weight=masks_weight,
    )
    assert ibot_loss.ndim == 0 and torch.isfinite(ibot_loss)

    hidden_loss, _ = HiddenLayerDistillationLoss()(
        [[torch.randn(2, 8, 384)]],
        [[torch.randn(2, 8, 384)]],
    )
    assert hidden_loss.ndim == 0 and torch.isfinite(hidden_loss)
