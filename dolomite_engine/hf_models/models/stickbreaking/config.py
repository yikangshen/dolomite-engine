from ...config import CommonConfig


class StickBreakingConfig(CommonConfig):
    model_type = "stickbreaking"

    def __init__(
        self,
        add_qkv_bias: bool = False,
        sb_remainder: bool = True,
        hir_gate: bool = False,
        router_aux_loss_coef: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sb_remainder = sb_remainder
        self.add_qkv_bias = add_qkv_bias
        self.hir_gate = hir_gate
        self.router_aux_loss_coef = router_aux_loss_coef

        if add_qkv_bias:
            assert not self.add_bias
