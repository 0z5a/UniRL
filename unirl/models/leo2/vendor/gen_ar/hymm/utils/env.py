import os
from typing import Optional

__USE_EXT_MODEL: Optional[bool] = None

def should_use_external_model(*, refresh: bool = False) -> bool:
    """
    根据环境变量 USE_EXTERNAL_TRANSFUSION_MODEL 决定是否使用外部模型。
    结果在首次调用后缓存；如需重新读取，请传 refresh=True。
    """
    global __USE_EXT_MODEL

    if refresh or __USE_EXT_MODEL is None:
        try:
            __USE_EXT_MODEL = bool(int(os.getenv("USE_EXTERNAL_TRANSFUSION_MODEL", "0")))
        except ValueError:
            __USE_EXT_MODEL = False

    return __USE_EXT_MODEL


def is_bitwise_align_mode() -> bool:
    """
    根据环境变量 PTM_TORCH_BITWISE_ALIGN_MODE 决定是否使用bitwise对齐模式。
    """
    return bool(int(os.getenv("PTM_TORCH_BITWISE_ALIGN_MODE", "0")))