from dataclasses import dataclass
import math, random

import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...enums import InitMethod
from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import StickBreakingConfig
from .layer import StickBreakingBlock
from ...config import CommonConfig
from ...modeling_utils import get_normalization_function, ParameterizedLinear

@dataclass
class BaseModelOutputWithPastAndAuxLoss(BaseModelOutputWithPast):
    aux_loss: torch.Tensor | None = None

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


class StickBreakingPreTrainedModel(PreTrainedModelMixin):
    config_class = StickBreakingConfig
    layer_class = StickBreakingBlock
    _no_split_modules = ["StickBreakingBlock"]


class StickBreakingModel(StickBreakingPreTrainedModel, BaseModelMixin):

    def __init__(self, config: CommonConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._init_model(config, **kwargs)

        if config.hir_gate:
            init_method = InitMethod(config.init_method)
            initializer_range = config.initializer_range
            m_width = config.m_width
            std = initializer_range
            if init_method == InitMethod.mup:
                std /= math.sqrt(m_width)
            self.hir_gate = nn.ModuleList(
                [
                    nn.Sequential(
                        get_normalization_function(
                            config.normalization_function, config.hidden_size, eps=config.layer_norm_epsilon),
                        ParameterizedLinear(config.hidden_size, 128, bias=False, std=std),
                        nn.GELU(),
                        ParameterizedLinear(128, 2, bias=True, std=std),
                        nn.LogSigmoid(),
                    )
                    for i in range(config.n_layer)
                ] # + [None] * (config.n_layer - config.n_layer // 2)
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

        sb_metadata = None

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
                    sb_metadata=sb_metadata,
                )
        else:
            hir_gate_list = []
            log_hir_gate = 0.
            aux_loss = 0
            for i in range(self.config.n_layer):
                block, hir_gate_fn = self.h[i], self.hir_gate[i]

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                log_beta = hir_gate_fn(hidden_states)
                # log_hir_gate = log_beta + log_hir_gate
                log_hir_gate = log_beta
                hir_gate = torch.exp(log_hir_gate)

                query_gate, key_gate = hir_gate.chunk(2, dim=-1)

                # key_mask = (key_gate > 0.05).to(query_gate.dtype)
                # key_gate = key_gate * key_mask

                # query_mask = (query_gate > 0.02).to(query_gate.dtype)
                # query_gate = query_gate * query_mask
                
                local_aux_loss = torch.mean(key_gate)
                aux_loss += local_aux_loss
                hir_gate_list.append((query_gate, key_gate))

                hidden_states = block(
                    hidden_states,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    rope_cos_sin=rope_cos_sin,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    sb_metadata=sb_metadata,
                    key_log_gate=(key_gate + 1e-6).log(),
                    query_gate=query_gate,
                )

                    
            if torch.distributed.get_rank() == 0 and random.random() < 0.1:
                for hir_gate in hir_gate_list:
                    print(
                        plot(hir_gate[0][:48].flatten().float().cpu().detach().numpy(), max_val=1), 
                        plot(hir_gate[1][:48].flatten().float().cpu().detach().numpy(), max_val=1)
                        )
                print("====================================")

        hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPastAndAuxLoss(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            aux_loss=aux_loss,
        )
