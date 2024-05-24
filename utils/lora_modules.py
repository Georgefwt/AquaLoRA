import torch
import torch.nn.functional as F
from diffusers.loaders import LoraLoaderMixin
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Union
from diffusers.models.lora import text_encoder_attn_modules, text_encoder_mlp_modules, PatchedLoraProjection
from diffusers.models.lora import LoRALinearLayer, LoRAConv2dLayer, LoRACompatibleConv, LoRACompatibleLinear

def CustomLoRALinearLayerforward(self, hidden_states: torch.Tensor, scale: float = 1.0):
    orig_dtype = hidden_states.dtype
    dtype = self.down.weight.dtype

    down_hidden_states = self.down(hidden_states.to(dtype))

    if isinstance(scale, torch.Tensor):
        mid = torch.diag_embed(scale)
        down_hidden_states = down_hidden_states @ mid

    up_hidden_states = self.up(down_hidden_states)

    if self.network_alpha is not None:
        up_hidden_states *= self.network_alpha / self.rank

    if isinstance(scale, float):
        return scale * up_hidden_states.to(orig_dtype)
    return up_hidden_states.to(orig_dtype)

def CustomLoRAConv2dLayerforward(self, hidden_states: torch.Tensor, scale: float = 1.0):
    orig_dtype = hidden_states.dtype
    dtype = self.down.weight.dtype

    down_hidden_states = self.down(hidden_states.to(dtype)) # [bz, channels, h, w]

    if isinstance(scale, torch.Tensor):
        down_hidden_states = down_hidden_states * scale[:, :, None, None]

    up_hidden_states = self.up(down_hidden_states)

    if self.network_alpha is not None:
        up_hidden_states *= self.network_alpha / self.rank

    if isinstance(scale, float):
        return scale * up_hidden_states.to(orig_dtype)
    return up_hidden_states.to(orig_dtype)

def CustomLoRACompatibleConvforward(self, hidden_states: torch.Tensor, scale: float = 1.0):
    if self.lora_layer is None:
        # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
        # see: https://github.com/huggingface/diffusers/pull/4315
        return F.conv2d(
            hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
    else:
        return super(LoRACompatibleConv, self).forward(hidden_states) + self.lora_layer(hidden_states, scale)

def CustomLoRACompatibleLinearforward(self, hidden_states: torch.Tensor, scale: float = 1.0):
    if self.lora_layer is None:
        out = super(LoRACompatibleLinear, self).forward(hidden_states)
        return out
    else:
        out = super(LoRACompatibleLinear, self).forward(hidden_states) + self.lora_layer(hidden_states, scale)
        return out


class CustomLoraLoaderMixin(LoraLoaderMixin):

    @classmethod
    def _modify_text_encoder(
        cls,
        text_encoder,
        lora_scale=1,
        network_alphas=None,
        rank: Union[Dict[str, int], int] = 4,
        dtype=None,
        patch_mlp=False,
        low_cpu_mem_usage=False,
    ):
        r"""
        Monkey-patches the forward passes of attention modules of the text encoder.
        """

        def create_patched_linear_lora(model, network_alpha, rank, dtype, lora_parameters):
            linear_layer = model.regular_linear_layer if isinstance(model, PatchedLoraProjection) else model
            ctx = nullcontext
            with ctx():
                model = PatchedLoraProjection(linear_layer, lora_scale, network_alpha, rank, dtype=dtype)

            lora_parameters.extend(model.lora_linear_layer.parameters())
            return model

        # First, remove any monkey-patch that might have been applied before
        cls._remove_text_encoder_monkey_patch_classmethod(text_encoder)

        lora_parameters = []
        network_alphas = {} if network_alphas is None else network_alphas
        is_network_alphas_populated = len(network_alphas) > 0

        for name, attn_module in text_encoder_attn_modules(text_encoder):
            query_alpha = network_alphas.pop(name + ".to_q_lora.down.weight.alpha", None)
            key_alpha = network_alphas.pop(name + ".to_k_lora.down.weight.alpha", None)
            value_alpha = network_alphas.pop(name + ".to_v_lora.down.weight.alpha", None)
            out_alpha = network_alphas.pop(name + ".to_out_lora.down.weight.alpha", None)

            if isinstance(rank, dict):
                current_rank = rank.pop(f"{name}.out_proj.lora_linear_layer.up.weight")
            else:
                current_rank = rank

            attn_module.q_proj = create_patched_linear_lora(
                attn_module.q_proj, query_alpha, current_rank, dtype, lora_parameters
            )
            attn_module.k_proj = create_patched_linear_lora(
                attn_module.k_proj, key_alpha, current_rank, dtype, lora_parameters
            )
            attn_module.v_proj = create_patched_linear_lora(
                attn_module.v_proj, value_alpha, current_rank, dtype, lora_parameters
            )
            attn_module.out_proj = create_patched_linear_lora(
                attn_module.out_proj, out_alpha, current_rank, dtype, lora_parameters
            )

        if patch_mlp:
            for name, mlp_module in text_encoder_mlp_modules(text_encoder):
                fc1_alpha = network_alphas.pop(name + ".fc1.lora_linear_layer.down.weight.alpha", None)
                fc2_alpha = network_alphas.pop(name + ".fc2.lora_linear_layer.down.weight.alpha", None)

                if isinstance(rank, dict):
                    current_rank_fc1 = rank.pop(f"{name}.fc1.lora_linear_layer.up.weight")
                    current_rank_fc2 = rank.pop(f"{name}.fc2.lora_linear_layer.up.weight")
                else:
                    current_rank_fc1 = rank
                    current_rank_fc2 = rank

                mlp_module.fc1 = create_patched_linear_lora(
                    mlp_module.fc1, fc1_alpha, current_rank_fc1, dtype, lora_parameters
                )
                mlp_module.fc2 = create_patched_linear_lora(
                    mlp_module.fc2, fc2_alpha, current_rank_fc2, dtype, lora_parameters
                )

        if is_network_alphas_populated and len(network_alphas) > 0:
            raise ValueError(
                f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
            )

        return lora_parameters
