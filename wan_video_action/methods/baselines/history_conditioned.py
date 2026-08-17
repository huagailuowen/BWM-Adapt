from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="history_conditioned_wm",
    display_name="History-Conditioned WM",
    family=MethodFamily.BASELINE,
    summary="Conditions query generation directly on raw support history.",
    training_protocol=(
        "Train the Wan backbone to consume the same support windows and query "
        "targets used by the other adaptive methods."
    ),
    inference_protocol=(
        "Encode K support trajectories as read-only history context and generate "
        "all disjoint queries without gradient-based test-time updates."
    ),
    adaptation_target=AdaptationTarget.HISTORY,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=True,
    invariants=("Support history and query trajectories remain disjoint.",),
)
