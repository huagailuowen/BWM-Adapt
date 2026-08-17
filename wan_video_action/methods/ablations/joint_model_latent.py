from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="joint_model_latent_training",
    display_name="Joint Model-Latent Training",
    family=MethodFamily.ABLATION,
    parent_slug="ours",
    summary="Tests whether alternating assimilation and consolidation are necessary.",
    training_protocol=(
        "Use the same grouped data, initialization, and fixed hardware-time budget "
        "as Ours, but update model and Z jointly "
        "instead of using separate frozen phases."
    ),
    inference_protocol="Use the same FP32 support-time Z optimization and read-only query protocol as Ours.",
    adaptation_target=AdaptationTarget.ENVIRONMENT_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=("Update scheduling is the only intended difference from Ours.",),
)
