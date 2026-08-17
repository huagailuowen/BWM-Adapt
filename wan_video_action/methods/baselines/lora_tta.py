from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="lora_tta",
    display_name="LoRA TTA",
    family=MethodFamily.BASELINE,
    summary="Support-time parameter adaptation from the Standard Pooled World Model.",
    training_protocol=(
        "Initialize from the normally trained Standard Pooled World Model checkpoint, "
        "not from Ours or a code-conditioned Stage 1 checkpoint."
    ),
    inference_protocol=(
        "Reset rank-8 LoRA to a shared zero initialization per environment, update "
        "it on K supports, freeze it, and reuse it for all disjoint queries."
    ),
    adaptation_target=AdaptationTarget.LORA,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=False,
    invariants=("Query loss and query tokens never update the adapted LoRA state at evaluation.",),
)
