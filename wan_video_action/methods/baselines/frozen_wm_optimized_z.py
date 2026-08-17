from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="frozen_wm_optimized_z",
    display_name="Frozen WM + Optimized Environment Code",
    family=MethodFamily.BASELINE,
    summary="Tests environment-code inference without continual shared-model consolidation.",
    training_protocol=(
        "Train the initial code-conditioned Wan model, freeze the shared Wan "
        "parameters, and optimize only environment codes as later environments arrive."
    ),
    inference_protocol=(
        "Use the same initialization, K supports, FP32 code optimizer, adaptation "
        "steps, and read-only query protocol as Ours."
    ),
    adaptation_target=AdaptationTarget.ENVIRONMENT_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=("Shared Wan parameters remain frozen after the initial training stage.",),
)
