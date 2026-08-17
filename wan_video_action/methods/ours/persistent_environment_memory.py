from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="ours",
    display_name="Ours",
    family=MethodFamily.OURS,
    summary="Persistent environment memory with alternating assimilation and consolidation.",
    training_protocol=(
        "Assign one persistent Z to each training environment and alternate "
        "environment-code optimization with shared Wan model updates over the "
        "progressive environment stream."
    ),
    inference_protocol=(
        "Initialize a fresh FP32 Z from the active-table mean, optimize only Z "
        "on K support trajectories, freeze it, and reuse it for every disjoint query."
    ),
    adaptation_target=AdaptationTarget.ENVIRONMENT_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=(
        "One training-time environment owns one persistent code.",
        "Support and query trajectories are disjoint.",
        "Query prediction does not update Z.",
    ),
)
