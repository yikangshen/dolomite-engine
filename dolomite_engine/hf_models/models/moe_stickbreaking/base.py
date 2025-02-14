import torch
import torch.nn as nn
from transformers import DynamicCache

from ...mixins import BaseMoEModelMixin, MoeModelOutputWithPastAndAuxLoss, PreTrainedMoEModelMixin
from .config import MoEStickBreakingConfig
from .layer import MoEStickBreakingBlock
from ....utils import divide_if_divisible
from ...config import CommonConfig
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import ParameterizedEmbedding, get_normalization_function


class MoEStickBreakingPreTrainedModel(PreTrainedMoEModelMixin):
    config_class = MoEStickBreakingConfig
    layer_class = MoEStickBreakingBlock
    _no_split_modules = ["MoEStickBreakingBlock"]


class MoEStickBreakingModel(MoEStickBreakingPreTrainedModel, BaseMoEModelMixin):
    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.m_emb = config.m_emb
        self.n_layer = config.n_layer
        self.initializer_range = config.initializer_range

        self.head_dim = divide_if_divisible(
            self.embed_dim,
            self.num_heads,
            f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.drop = nn.Identity() if config.embd_pdrop == 0 else nn.Dropout(config.embd_pdrop)
        self.h = self.layer_class(
            config,
            attention_implementation=self.attention_implementation,
            use_padding_free_transformer=self._use_padding_free_transformer,
            moe_implementation=self.moe_implementation,
            layer_idx=None,
        )
        self.ln_f = get_normalization_function(
            config.normalization_function, self.embed_dim, eps=config.layer_norm_epsilon
        )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()

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
        output_router_logits: bool | None = None,
        output_aux_loss: bool = True,
    ) -> MoeModelOutputWithPastAndAuxLoss:
        (
            output_hidden_states,
            use_cache,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
            output_router_logits,
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
            output_router_logits=output_router_logits,
        )

        sb_metadata = None
        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None
        total_aux_loss = 0.0

        for i in range(self.n_layer):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            outputs = self.h(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                output_router_logits=output_router_logits,
                output_aux_loss=output_aux_loss,
                sb_metadata=sb_metadata,
            )

            hidden_states = outputs[0]
            outputs = outputs[1:]

            if output_router_logits:
                all_router_logits += (outputs[0],)
                outputs = outputs[1:]

            if output_aux_loss:
                total_aux_loss += outputs[0]

        hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return MoeModelOutputWithPastAndAuxLoss(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            router_logits=all_router_logits,
            aux_loss=total_aux_loss / self.n_layer,
        )
