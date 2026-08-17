from ..protocol import AdaptationTarget, MethodFamily, MethodSpec, QueryStatePolicy


SPEC = MethodSpec(
    slug="per_trajectory_z",
    display_name="Per-Trajectory Latent",
    family=MethodFamily.ABLATION,
    parent_slug="ours",
    summary="Removes environment-level sharing by assigning one code per trajectory.",
    training_protocol=(
        "Match Ours in data protocol, latent dimension, and fixed hardware-time budget, "
        "but give each training trajectory an independent persistent Z."
    ),
    inference_protocol="Adapt one fresh Z on the support set and freeze it for disjoint query prediction.",
    adaptation_target=AdaptationTarget.ENVIRONMENT_CODE,
    query_state_policy=QueryStatePolicy.READ_ONLY,
    requires_grouped_training=False,
    invariants=("Latent scope is the only intended difference from Ours.",),
)
