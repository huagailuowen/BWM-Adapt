from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="shuffled_environment_grouping",
    display_name="Shuffled Environment Grouping",
    family=MethodFamily.ABLATION,
    parent_slug="ours",
    summary="Tests whether correct environment grouping supplies the useful latent signal.",
    training_protocol=(
        "Match Ours exactly but replace true environment groups with one fixed, "
        "seeded shuffled assignment that preserves group sizes and resource budget."
    ),
    inference_protocol="Use the same FP32 support-time Z optimization and read-only query protocol as Ours.",
    adaptation_target=AdaptationTarget.ENVIRONMENT_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=("Only the environment-to-group assignment changes relative to Ours.",),
)
