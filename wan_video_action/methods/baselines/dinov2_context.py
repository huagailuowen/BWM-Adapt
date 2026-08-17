from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="dinov2_amortized_context",
    display_name="DINOv2 Amortized Context Encoder",
    family=MethodFamily.BASELINE,
    summary="Amortizes environment-code inference from support video and action chunks.",
    training_protocol=(
        "Use frozen DINOv2-B/14 on eight sparse frames spanning the same full "
        "support window, encode every intervening action chunk, project each "
        "K=1 support trajectory to 32-D Z, and optimize disjoint query "
        "flow-matching losses without exposing query pixels to the encoder."
    ),
    inference_protocol="Infer one Z in a single encoder forward pass and reuse it for all disjoint queries.",
    adaptation_target=AdaptationTarget.AMORTIZED_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=(
        "Wan initialization, environment stream, and fixed hardware-time budget match Ours.",
        "The eight frames span the complete support window rather than defining an eight-frame crop.",
    ),
)
