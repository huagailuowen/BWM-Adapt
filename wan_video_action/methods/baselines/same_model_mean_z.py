from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="same_model_mean_z",
    display_name="Same-Model Mean-Z",
    family=MethodFamily.BASELINE,
    summary="No-adaptation control using the same architecture and checkpoint as Ours.",
    training_protocol="Reuse the complete trained checkpoint and active environment-code table from Ours.",
    inference_protocol=(
        "Use the mean active training-time Z for every query and perform no "
        "support-time optimization."
    ),
    adaptation_target=AdaptationTarget.ENVIRONMENT_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=("Architecture, weights, code mean, sampler, and query inputs match Ours.",),
)
