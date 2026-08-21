from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


NEW_CONTEXT_200_THEN_JOINT_800 = MethodSpec(
    slug="new_context_200_then_joint_800",
    display_name="New-Context 200 Then Joint 800",
    family=MethodFamily.ABLATION,
    parent_slug="ours",
    summary="Separates new-code acquisition from the remaining joint curriculum updates.",
    training_protocol=(
        "At each curriculum wave, freeze the model and optimize only newly activated "
        "environment codes for 200 steps, then jointly optimize the model and all active "
        "codes for the remaining 800 steps."
    ),
    inference_protocol="Use the same FP32 support-time Z optimization and read-only query protocol as Ours.",
    adaptation_target=AdaptationTarget.ENVIRONMENT_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=("Only the curriculum update schedule differs from Ours.",),
)


ALL_ACTIVE_JOINT_NO_CURRICULUM = MethodSpec(
    slug="all_active_joint_no_curriculum",
    display_name="All-Active Joint Training Without Curriculum",
    family=MethodFamily.ABLATION,
    parent_slug="ours",
    summary="Tests whether progressive environment activation is necessary.",
    training_protocol=(
        "Activate all training environments at step 1 and jointly optimize the model and "
        "all environment codes for the full training budget, without frozen phases."
    ),
    inference_protocol="Use the same FP32 support-time Z optimization and read-only query protocol as Ours.",
    adaptation_target=AdaptationTarget.ENVIRONMENT_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=("All training environments are active from the first update.",),
)

