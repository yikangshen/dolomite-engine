from ...config import CommonConfig


class RNNDolomiteConfig(CommonConfig):
    model_type = "rnn_dolomite"

    def __init__(
        self,
        attention_patterns: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.attention_patterns = attention_patterns
        assert len(attention_patterns) == self.n_layer, "Attention patterns must be specified for each layer"
