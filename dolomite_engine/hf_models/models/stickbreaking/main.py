from dataclasses import dataclass
from ...config import CommonConfig
from ...mixins import CausalLMModelMixin
from ...loss import get_autoregressive_language_modeling_loss
from .base import StickBreakingModel, StickBreakingPreTrainedModel, BaseModelOutputWithPastAndAuxLoss
import torch
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputWithPastAndAuxLoss(CausalLMOutputWithPast):
    aux_loss: torch.Tensor | None = None


class StickBreakingForCausalLM(StickBreakingPreTrainedModel, CausalLMModelMixin):
    def __init__(self, config: CommonConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.router_aux_loss_coef = config.router_aux_loss_coef

    base_model_class = StickBreakingModel
    def forward(
        self,
        input_ids: torch.Tensor | list[list[int]] | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | list[list[int]] | None = None,
        position_ids: torch.Tensor | list[list[int]] | None = None,
        inputs_embeds: torch.Tensor | list[list[float]] | None = None,
        labels: torch.Tensor | list[list[int]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> CausalLMOutputWithPastAndAuxLoss:
        needs_eval_fix = self._use_padding_free_transformer and cu_seqlens is None 
        if needs_eval_fix:
            assert input_ids.size(0) == 1, input_ids.size()
            cu_seqlens = torch.tensor([0, input_ids.size(1)],
                                      device=input_ids.device,
                                      dtype=input_ids.dtype)
            max_seqlen = input_ids.size(1)
            position_ids = torch.arange(max_seqlen, device=input_ids.device, dtype=input_ids.dtype)
            input_ids = input_ids[0]

        input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen = self.prepare_inputs_for_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # ==========================================================================================
        # padding_free:
        #     input_ids -> (total_q)
        #     attention_mask -> None
        #     position_ids -> (total_q)
        # else:
        #     input_ids -> (batch_size, query_length)
        #     attention_mask -> None or (batch_size, key_length)
        #     position_ids -> None or (batch_size, key_length)
        # ==========================================================================================

        transformer_outputs: BaseModelOutputWithPastAndAuxLoss = self.transformer(
            input_ids,
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

        lm_logits = self.get_lm_logits(transformer_outputs.last_hidden_state)

        if self.m_width is not None:
            lm_logits = lm_logits / self.m_width

        lm_loss = None
        if labels is not None:
            lm_loss = get_autoregressive_language_modeling_loss(
                lm_logits=lm_logits,
                labels=labels,
                upcast_logits_for_loss=self.upcast_logits_for_loss,
                cu_seqlens=cu_seqlens,
                use_padding_free_transformer=self._use_padding_free_transformer,
                reduction=reduction,
            )

        aux_loss = transformer_outputs.aux_loss

        if lm_loss is None:
            loss = None
        else:
            loss = lm_loss + self.router_aux_loss_coef * aux_loss

        if needs_eval_fix:
            lm_logits = lm_logits[None, :]

        return CausalLMOutputWithPastAndAuxLoss(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            aux_loss=transformer_outputs.aux_loss,
        )