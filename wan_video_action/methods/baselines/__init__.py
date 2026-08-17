from .dinov2_context import SPEC as DINOV2_AMORTIZED_CONTEXT
from .standard_pooled_wm import SPEC as STANDARD_POOLED_WM
from .frozen_wm_optimized_z import SPEC as FROZEN_WM_OPTIMIZED_Z
from .history_conditioned import SPEC as HISTORY_CONDITIONED_WM
from .lora_tta import SPEC as LORA_TTA
from .same_model_mean_z import SPEC as SAME_MODEL_MEAN_Z
from .ttt_kqv import SPEC as TTT_KQV

SPECS = (
    STANDARD_POOLED_WM,
    SAME_MODEL_MEAN_Z,
    HISTORY_CONDITIONED_WM,
    LORA_TTA,
    TTT_KQV,
    DINOV2_AMORTIZED_CONTEXT,
    FROZEN_WM_OPTIMIZED_Z,
)

__all__ = [
    "DINOV2_AMORTIZED_CONTEXT",
    "STANDARD_POOLED_WM",
    "FROZEN_WM_OPTIMIZED_Z",
    "HISTORY_CONDITIONED_WM",
    "LORA_TTA",
    "SAME_MODEL_MEAN_Z",
    "SPECS",
    "TTT_KQV",
]
