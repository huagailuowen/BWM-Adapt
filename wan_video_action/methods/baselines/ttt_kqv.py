from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="ttt_kqv",
    display_name="TTT-KQV",
    family=MethodFamily.BASELINE,
    summary="Implicit environment memory stored in support-written attention fast weights.",
    training_protocol=(
        "Process K supports sequentially from a learned shared fast-state "
        "initialization. Query flow-matching loss backpropagates through the "
        "support-time write process, while query tokens do not write fast weights."
    ),
    inference_protocol=(
        "Write the support set once, freeze the resulting fast state, generate all "
        "queries by fixed-state reads, and reset before the next environment."
    ),
    adaptation_target=AdaptationTarget.FAST_WEIGHTS,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=(
        "Fast state carries across support trajectories only.",
        "Temporal positions reset at trajectory boundaries.",
        "Query tokens are read-only during outer training and evaluation.",
    ),
)
