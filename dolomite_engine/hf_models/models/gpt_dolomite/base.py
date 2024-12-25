import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import GPTDolomiteConfig
from .layer import GPTDolomiteBlock
from ...config import CommonConfig
from ...enums import InitMethod
from ...modeling_utils import get_normalization_function, ParameterizedLinear

# coding=utf-8
import numpy as np
chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


class BarHack(str):

    def __str__(self):
        return self.internal

    def __len__(self):
        return 1


def plot(arr, max_val=None):
    if max_val is None:
        max_arr = arr
        max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))

    opts = np.get_printoptions()
    np.set_printoptions(edgeitems=500)
    fig = np.array2string(arr,
                          formatter={
                              'float_kind': lambda x: visual(x, max_val),
                              'int_kind': lambda x: visual(x, max_val)},
                          max_line_width=5000
                          )
    np.set_printoptions(**opts)

    return fig


def visual(val, max_val):
    val = np.clip(val, 0, max_val)
    if abs(val) == max_val:
        step = len(chars) - 1
    else:
        step = int(abs(float(val) / max_val) * len(chars))
    colourstart = ""
    colourend = ""
    if val < 0:
        colourstart, colourend = '\033[90m', '\033[0m'
    return colourstart + chars[step] + colourend


class GPTDolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = GPTDolomiteConfig
    layer_class = GPTDolomiteBlock
    _no_split_modules = ["GPTDolomiteBlock"]


class GPTDolomiteModel(GPTDolomitePreTrainedModel, BaseModelMixin):

    def __init__(self, config: CommonConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._init_model(config, **kwargs)

        if config.hir_gate:
            init_method = InitMethod(config.init_method)
            initializer_range = config.initializer_range
            m_width = config.m_width
            
            # std = initializer_range
            # if init_method == InitMethod.mup:
            #     std /= math.sqrt(m_width)
            std = 0
            self.hir_gate = nn.ModuleList(
                [
                    nn.Sequential(
                        get_normalization_function(
                            config.normalization_function, config.hidden_size, eps=config.layer_norm_epsilon),
                        ParameterizedLinear(config.hidden_size, 1, bias=True, std=std),
                        nn.Sigmoid(),
                    )
                    for i in range(config.n_layer // 2)
                ]
            )

    
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        (
            output_hidden_states,
            use_cache,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # ==========================================================================================
        # padding_free:
        #     attention_mask -> None
        # flash:
        #     attention_mask -> (batch_size, key_length)
        # else:
        #     attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
        all_hidden_states = () if output_hidden_states else None

        if not hasattr(self, "hir_gate"):
            for block in self.h:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                hidden_states = block(
                    hidden_states,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    rope_cos_sin=rope_cos_sin,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
        else:
            hir_gate_list = []
            hir_gate = 1
            for i in range(self.config.n_layer // 2):
                block, hir_gate_fn = self.h[i], self.hir_gate[i]

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                hir_gate = hir_gate_fn(hidden_states) * hir_gate
                hir_gate_list.append(hir_gate)

                print(plot(hir_gate.flatten()[:50].float().detach().cpu().numpy(), max_val=1))

                hidden_states = block(
                    hidden_states,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    rope_cos_sin=rope_cos_sin,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    hir_gate=hir_gate,
                )

            for i in range(self.config.n_layer // 2, self.config.n_layer):
                block = self.h[i]

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                hir_gate = hir_gate_list[self.config.n_layer - i - 1]

                hidden_states = block(
                    hidden_states,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    rope_cos_sin=rope_cos_sin,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    hir_gate=hir_gate,
                )

        hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )