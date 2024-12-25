from ...config import CommonConfig


class GPTDolomiteConfig(CommonConfig):
    model_type = "gpt_dolomite"

    def __init__(
        self,
        hir_gate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hir_gate = hir_gate
