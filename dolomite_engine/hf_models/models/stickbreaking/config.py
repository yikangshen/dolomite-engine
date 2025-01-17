from ...config import CommonConfig


class StickBreakingConfig(CommonConfig):
    model_type = "stickbreaking"

    def __init__(
        self,
        add_qkv_bias: bool = False,
        sb_remainder: bool = True,
        hir_gate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sb_remainder = sb_remainder
        self.add_qkv_bias = add_qkv_bias
        self.hir_gate = hir_gate

        if add_qkv_bias:
            assert not self.add_bias
