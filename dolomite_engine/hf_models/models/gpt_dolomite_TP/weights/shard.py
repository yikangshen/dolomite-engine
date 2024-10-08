import torch

from .....utils import ProcessGroupManager, SafeTensorsWeightsManager
from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils import is_glu
from ....modeling_utils_TP import get_tensor_parallel_vocab_info, tensor_parallel_split_safetensor_slice
from ....utils import divide_if_divisible
from ...gpt_dolomite import GPTDolomiteConfig


def get_gpt_dolomite_tp_state_dict(
    config: GPTDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    tensor_parallel_word_embeddings: bool,
) -> dict:
    # word embeddings
    state_dict = _get_word_embedding_weights(
        safetensors_weights_manager,
        prefix="transformer.wte.",
        vocab_size=config.vocab_size,
        tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
    )

    # positional embeddings
    if PositionEmbeddingType(config.position_embedding_type) == PositionEmbeddingType.learned_absolute:
        state_dict.update(
            _get_word_embedding_weights(
                safetensors_weights_manager,
                prefix="transformer.wpe.",
                vocab_size=config.n_positions,
                tensor_parallel_word_embeddings=False,
            )
        )

    for layer_idx in range(config.n_layer):
        prefix = f"transformer.h.{layer_idx}."

        state_dict.update({prefix + "ln_1.weight": safetensors_weights_manager.get_tensor(prefix + "ln_1.weight")})
        if safetensors_weights_manager.has_tensor(prefix + "ln_1.bias"):
            state_dict.update({prefix + "ln_1.bias": safetensors_weights_manager.get_tensor(prefix + "ln_1.bias")})

        state_dict.update(
            _get_attention_weights(
                config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "attn."
            )
        )

        state_dict.update({prefix + "ln_2.weight": safetensors_weights_manager.get_tensor(prefix + "ln_2.weight")})
        if safetensors_weights_manager.has_tensor(prefix + "ln_2.bias"):
            state_dict.update({prefix + "ln_2.bias": safetensors_weights_manager.get_tensor(prefix + "ln_2.bias")})

        state_dict.update(
            _get_mlp_weights(
                config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "mlp."
            )
        )

    state_dict.update({"transformer.ln_f.weight": safetensors_weights_manager.get_tensor("transformer.ln_f.weight")})
    if safetensors_weights_manager.has_tensor("transformer.ln_f.bias"):
        state_dict.update({"transformer.ln_f.bias": safetensors_weights_manager.get_tensor("transformer.ln_f.bias")})

    if not config.tie_word_embeddings:
        state_dict.update(
            _get_word_embedding_weights(
                safetensors_weights_manager=safetensors_weights_manager,
                prefix="lm_head.",
                vocab_size=config.vocab_size,
                tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
            )
        )

    return state_dict


def _get_word_embedding_weights(
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
    vocab_size: int,
    tensor_parallel_word_embeddings: bool,
) -> dict:
    if tensor_parallel_word_embeddings:
        vocab_start_index, vocab_end_index, vocab_size_per_tensor_parallel_rank = get_tensor_parallel_vocab_info(
            vocab_size
        )

        weight = safetensors_weights_manager.get_slice(prefix + "weight")[vocab_start_index:vocab_end_index, :]

        if weight.shape[0] < vocab_size_per_tensor_parallel_rank:
            weight = torch.cat(
                [
                    weight,
                    torch.zeros((vocab_size_per_tensor_parallel_rank - weight.shape[0], weight.shape[1])),
                ]
            )
    else:
        weight = safetensors_weights_manager.get_tensor(prefix + "weight")

    return {prefix + "weight": weight}


def _get_attention_weights(
    config: GPTDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
) -> None:
    state_dict = {}

    if AttentionHeadType(config.attention_head_type) == AttentionHeadType.mqa:
        tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        global_hidden_size = config.n_embd
        head_dim = divide_if_divisible(global_hidden_size, config.n_head, "")

        hidden_size_per_rank = divide_if_divisible(global_hidden_size, tp_world_size, "")
        start_index = tp_rank * hidden_size_per_rank
        end_index = (tp_rank + 1) * hidden_size_per_rank

        weight = safetensors_weights_manager.get_slice(prefix + "c_attn.weight")
        state_dict[prefix + "c_attn.q_attn.weight"] = weight[start_index:end_index, :]
        state_dict[prefix + "c_attn.kv_attn.weight"] = weight[
            global_hidden_size : global_hidden_size + 2 * head_dim, :
        ]

        if config.add_bias:
            bias = safetensors_weights_manager.get_slice(prefix + "c_attn.bias")
            state_dict[prefix + "c_attn.q_attn.bias"] = bias[start_index:end_index]
            state_dict[prefix + "c_attn.kv_attn.bias"] = bias[global_hidden_size : global_hidden_size + 2 * head_dim]
    else:
        state_dict.update(
            _get_column_parallel_weights(
                config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "c_attn."
            )
        )

    state_dict.update(
        _get_row_parallel_weights(
            config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "c_proj."
        )
    )

    return state_dict


def _get_mlp_weights(
    config: GPTDolomiteConfig,
    safetensors_weights_manager: SafeTensorsWeightsManager,
    prefix: str,
) -> None:
    # GLU is a special case and needs to be handled explicitely
    if is_glu(config.activation_function):
        weight = safetensors_weights_manager.get_slice(prefix + "c_fc.weight")

        tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        shape = weight.get_shape()
        stride = divide_if_divisible(
            shape[0],
            tp_world_size * 2,
            f"split dimension ({0}) is not divisible by 2 x tensor parallel world size (2 x {tp_world_size})",
        )

        # split weight tensors into gate and non-gate
        start_end = (tp_rank * stride, (tp_rank + 1) * stride)
        weight_1 = tensor_parallel_split_safetensor_slice(weight, 0, start_end)
        if config.add_bias:
            bias = safetensors_weights_manager.get_slice(prefix + "c_fc.bias")
            bias_1 = tensor_parallel_split_safetensor_slice(bias, 0, start_end)

        start_end = (
            (tp_world_size + tp_rank) * stride,
            (tp_world_size + tp_rank + 1) * stride,
        )
        weight_2 = tensor_parallel_split_safetensor_slice(weight, 0, start_end)
        if config.add_bias:
            bias_2 = tensor_parallel_split_safetensor_slice(bias, 0, start_end)

        state_dict = {prefix + "c_fc.weight": torch.cat([weight_1, weight_2])}
        if config.add_bias:
            state_dict[prefix + "c_fc.bias"] = torch.cat([bias_1, bias_2])
    else:
        state_dict = _get_column_parallel_weights(
            config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "c_fc."
        )

    state_dict.update(
        _get_row_parallel_weights(
            config=config, safetensors_weights_manager=safetensors_weights_manager, prefix=prefix + "c_proj."
        )
    )

    return state_dict


def _get_column_parallel_weights(
    config: GPTDolomiteConfig, safetensors_weights_manager: SafeTensorsWeightsManager, prefix: str
) -> dict:
    weight = safetensors_weights_manager.get_slice(prefix + "weight")
    weight = tensor_parallel_split_safetensor_slice(weight, dim=0)
    state_dict = {prefix + "weight": weight}

    if config.add_bias:
        bias = safetensors_weights_manager.get_slice(prefix + "bias")
        bias = tensor_parallel_split_safetensor_slice(bias, dim=0)
        state_dict[prefix + "bias"] = bias

    return state_dict


def _get_row_parallel_weights(
    config: GPTDolomiteConfig, safetensors_weights_manager: SafeTensorsWeightsManager, prefix: str
) -> dict:
    weight = safetensors_weights_manager.get_slice(prefix + "weight")
    weight = tensor_parallel_split_safetensor_slice(weight, dim=1)
    state_dict = {prefix + "weight": weight}

    if config.add_bias:
        state_dict[prefix + "bias"] = safetensors_weights_manager.get_tensor(prefix + "bias")

    return state_dict
