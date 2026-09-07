from mmengine.dist import get_dist_info
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS


def _vit_layer_id(name, num_max_layer):
    if name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    if name.startswith("backbone.patch_embed") or ".patch_embed" in name:
        return 0
    if name.startswith("backbone.blocks"):
        return int(name.split(".")[2]) + 1
    if ".blocks." in name:
        return int(name[name.find(".blocks.") :].split(".")[2]) + 1
    return num_max_layer - 1


def _dinov3_layer_id(name, num_layers):
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name or ".rope_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    return layer_id


def _dinov3_lr_scale(name, layer_decay_rate=1.0, num_layers=12):
    layer_id = _dinov3_layer_id(name, num_layers)
    return layer_decay_rate ** (num_layers + 1 - layer_id)


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor_DINOv3(DefaultOptimWrapperConstructor):
    """DINOv3-style layer decay used by the previous in-house fine-tuning experiments."""

    def add_params(self, params, module, prefix="", is_dcn_module=None):
        parameter_groups = {}
        num_layers = self.paramwise_cfg.get("num_layers")
        layer_decay_rate = self.paramwise_cfg.get("layer_decay_rate")
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name:
                decay_type = "no_decay"
                this_weight_decay = 0.0
            else:
                decay_type = "decay"
                this_weight_decay = weight_decay

            layer_id = _dinov3_layer_id(name, num_layers)
            group_name = f"layer_{layer_id}_{decay_type}"
            if group_name not in parameter_groups:
                scale = _dinov3_lr_scale(name, layer_decay_rate=layer_decay_rate, num_layers=num_layers)
                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            print(f"Build DINOv3LayerDecayOptimizerConstructor {layer_decay_rate:f} - {num_layers:d}")
        params.extend(parameter_groups.values())


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor_ViT(DefaultOptimWrapperConstructor):
    """ViT layer decay constructor from the previous in-house fine-tuning code."""

    def add_params(self, params, module, prefix="", is_dcn_module=None):
        parameter_groups = {}
        num_layers = self.paramwise_cfg.get("num_layers") + 2
        layer_decay_rate = self.paramwise_cfg.get("layer_decay_rate")
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name:
                decay_type = "no_decay"
                this_weight_decay = 0.0
            else:
                decay_type = "decay"
                this_weight_decay = weight_decay

            layer_id = _vit_layer_id(name, num_layers)
            group_name = f"layer_{layer_id}_{decay_type}"
            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)
                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            print(f"Build LayerDecayOptimizerConstructor {layer_decay_rate:f} - {num_layers:d}")
        params.extend(parameter_groups.values())
