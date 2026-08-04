# Real and Counterfactual Multi-C Bridge Training

Status: design specification, 2026-07-28.

This document records the current agreed design. It supersedes earlier proposals
that allowed one counterfactual base case to train all four endpoint rays.

## 1. Objective

The purpose of bridge training is to create a useful conditional flow field
between a fixed Global latent and four learned environment endpoint latents.
This should let Stage2 start near Global and move toward the endpoint matching
the observed environment.

Training uses two parallel data paths:

- 60% ordinary real-data flow matching.
- 40% counterfactual correction.

The ratio is defined at the normalized training-block or DiT-forward-example
level. A counterfactual example containing four target losses still counts as
one counterfactual example, not four examples.

## 2. Bridge geometry

For endpoint environment \(e\), sample a raw path position \(s\in[0,1]\):

\[
q(s)=1-(1-s)^5
\]

\[
C_e(s)=(1-q(s))C_G+q(s)C_e
\]

The desired environment mixture at this position is:

\[
w_e(s)=\frac14+\frac34q(s)
\]

\[
w_{k\neq e}(s)=\frac{1-q(s)}4
\]

At exact Global, all four environments have weight \(1/4\). At endpoint \(e\),
environment \(e\) has weight 1 and the others have weight 0.

For every \(s>0\), the real endpoint is the largest individual target. It does
not necessarily exceed the sum of all three fake targets very close to Global.
With the current curve, it exceeds their combined weight at approximately
\(s>0.078\).

The non-Global C sampling target remains:

- 40% endpoint.
- 30% near-Global.
- 30% interior.

Near-Global samples are necessary for constructing the Stage2 gradient field
around its initialization.

## 3. Real-data path

### 3.1 Data requirement

A real-data macro-batch must contain balanced chunks from all four
environments. Each chunk uses its own real video, action, history, and initial
state.

For environment \(e\), chunk \(i\):

\[
z_{e,i}=\operatorname{VAE}(video_{e,i})
\]

\[
x_{e,i,\sigma}=(1-\sigma)z_{e,i}+\sigma\epsilon_{e,i}
\]

\[
v^*_{e,i}=\epsilon_{e,i}-z_{e,i}
\]

The clean latent, noise, noised latent, action condition, and flow target are
independent of C and should be prepared once and reused.

### 3.2 C coverage

The same balanced real batch may supervise many C positions on all four rays:

\[
C_G\rightarrow C_A,\quad
C_G\rightarrow C_B,\quad
C_G\rightarrow C_C,\quad
C_G\rightarrow C_D
\]

For sampled condition \(C_j\), the model prediction is:

\[
\hat v_{e,i,j}
=v_\theta(x_{e,i,\sigma},\sigma,a_{e,i},C_j)
\]

The weighted real loss at \(C_j\) is:

\[
L_{\mathrm{real}}(C_j)
=
\sum_e w_e(C_j)
\frac{1}{N_e}
\sum_i
\left\|\hat v_{e,i,j}-v^*_{e,i}\right\|^2
\]

### 3.3 Important computation boundary

Latent preparation can be reused across C positions, but the DiT prediction
cannot. Because the model is conditioned on C:

\[
v_\theta(x,C_1)\neq v_\theta(x,C_2)
\]

It is invalid to calculate scalar losses at one C and represent other C
positions by only changing the environment weights. Each C requires its own
C-conditioned DiT evaluation.

Efficiency comes from batching these evaluations, not from reusing one model
prediction.

### 3.4 Vectorized layout

Use C as an additional batch axis:

\[
[B,E,K,\ldots]\rightarrow[B\times E\times K,\ldots]
\]

Here \(B\) is chunks per environment, \(E=4\), and \(K\) is the number of C
positions.

The implementation should:

- Encode each real video once.
- Reuse noised latents and flow targets across C.
- Expand physical-condition tokens over K conditions.
- Flatten `(chunk, environment, C)` for the DiT forward.
- Restore `[K,E,B,...]` before weighted reduction.
- Split K into microbatches when memory is insufficient.
- Perform one optimizer step only after all intended C blocks are accumulated.

## 4. Counterfactual path

### 4.1 Smallest counterfactual unit

Start from one real chunk belonging to environment \(e\). Preserve the same
initial state, history, and action, and use the frozen Teacher to construct the
other three endpoint outcomes:

\[
Z_b=
\left\{
z_b^{(A)},z_b^{(B)},z_b^{(C)},z_b^{(D)}
\right\}
\]

Exactly one version, \(z_b^{(e)}\), is real. The other three are Teacher
counterfactual videos.

The four versions should be generated offline. Their VAE latents should be
cached so that bridge training does not repeatedly run the Teacher or VAE.

### 4.2 Source sampling

Choose the source uniformly:

\[
u\sim\operatorname{Uniform}(A,B,C,D)
\]

Thus the real source and each fake source have probability \(1/4\). A balanced
shuffled permutation may be used instead of independent random draws to reduce
finite-batch variance while preserving the same distribution.

Create the high-noise input:

\[
x_\sigma=(1-\sigma)z_b^{(u)}+\sigma\epsilon
\]

### 4.3 Allowed C positions

A counterfactual unit from real environment \(e\) may train only:

\[
C_G\rightarrow C_e
\]

It must not train the other three endpoint rays. Allowing a base case from
environment \(e\) to train the ray toward endpoint \(r\neq e\) would make a
Teacher-generated fake video define the dominant endpoint behavior.

Across a balanced macro-batch:

- Real A cases cover the Global-to-A ray.
- Real B cases cover the Global-to-B ray.
- Real C cases cover the Global-to-C ray.
- Real D cases cover the Global-to-D ray.

The complete batch therefore covers all four rays, while each individual
counterfactual unit remains anchored to its real environment.

### 4.4 Multi-target correction loss

At \(C_e(s)\), perform one DiT forward:

\[
\hat v
=v_\theta(x_\sigma,\sigma,a_b,C_e(s))
\]

Construct one velocity target for every endpoint video:

\[
v_k^*=\frac{x_\sigma-z_b^{(k)}}{\sigma}
\]

The counterfactual loss is:

\[
L_{\mathrm{CF}}(C_e(s))
=
\sum_{k\in\{A,B,C,D\}}
w_k(C_e(s))
\left\|\hat v-v_k^*\right\|^2
\]

This uses one expensive DiT forward and four inexpensive target comparisons.

For gradient computation, define:

\[
\bar z_b(C)=\sum_k w_k(C)z_b^{(k)}
\]

\[
\bar v^*=\frac{x_\sigma-\bar z_b(C)}{\sigma}
\]

Training against \(\bar v^*\) produces the same model-parameter gradient as the
four weighted MSE terms. The exact four losses must still be calculated for
logging because they expose Teacher bias and real-versus-fake behavior.

### 4.5 Multi-C reuse

One four-video counterfactual unit may be reused at many C positions and
timesteps along its allowed real-environment ray.

For each `(base chunk, C, timestep)`:

- Select one of the four sources.
- Construct one noised source latent.
- Execute one C-conditioned DiT forward.
- Compare the prediction with all four weighted targets.

Different C positions still require separate DiT evaluations. They should be
flattened into the batch dimension or processed as gradient-accumulated
microbatches.

## 5. Exact Global policy

The numeric Global context is trained during its initial warmup and then
frozen.

Exact Global should be maintained using real-data replay only:

- Counterfactual sampling uses \(s>0\).
- Exact \(s=0\) receives no counterfactual examples.
- Real examples preserve the model's functional behavior under the fixed
  Global condition.

This prevents exact Global from being dominated by the three fake targets in
each counterfactual unit.

## 6. Timestep policy

Real flow matching uses the normal full timestep distribution.

Counterfactual correction uses the agreed high-noise distribution:

- 20% noise fraction in `[0.90, 1.00]`.
- 60% noise fraction in `[0.70, 0.90]`.
- 20% noise fraction in `[0.55, 0.70]`.

One macro-batch should contain multiple timestep buckets. Samples inside a
comparison block may share a timestep and noise construction, while different
blocks use independently sampled timesteps.

Avoid the full Cartesian product of every C and every timestep. Assign a small
number of stratified timestep buckets across C blocks to control DiT cost.

## 7. 60/40 optimizer update

Each optimizer update should accumulate five normalized blocks:

- Three real-data blocks.
- Two counterfactual blocks.

After all five blocks:

\[
L_{\mathrm{total}}
=0.6\operatorname{mean}(L_{\mathrm{real}})
+0.4\operatorname{mean}(L_{\mathrm{CF}})
\]

Then execute one gradient clipping operation and one optimizer step.

The two branches must be normalized independently before applying the 0.6 and
0.4 coefficients. Four target terms inside one counterfactual block must not
silently multiply its effective weight by four.

## 8. Freeze and training policy

During initial Global warmup:

- Train the numeric Global context.
- Freeze endpoint contexts.
- Freeze the model parameters intended to remain anchored.

After Global warmup:

- Freeze numeric Global.
- Freeze endpoint context values.
- Train the selected model parameters that construct the conditional vector
  field.
- Preserve exact Global behavior through real replay.
- Preserve endpoint behavior through the real-data branch and endpoint
  sampling.

Any future decision to unfreeze endpoint contexts must be explicit and logged.
Intermediate bridge points are generated values, not independent persistent
parameters.

## 9. Required logging

At minimum, log:

- Normalized real loss.
- Normalized counterfactual loss.
- Total `0.6/0.4` loss.
- Per-environment real loss.
- Per-ray and path-position-bin loss.
- Exact endpoint real replay loss.
- Near-Global real loss.
- Counterfactual real-target component.
- Counterfactual aggregate fake-target component.
- Each of the four endpoint target components.
- Real-source versus fake-source loss.
- Source environment and real environment.
- C path endpoint, raw position \(s\), and curved position \(q(s)\).
- Timestep and noise-fraction band.
- Number of distinct C positions and timestep buckets.
- Teacher-versus-real target disagreement.
- Endpoint and Global retention metrics.

## 10. Invariants

- Do not add lamp-region weighting or any manually emphasized image region.
- Do not add a VACE branch, editing branch, source token, or task token.
- Teacher videos are detached offline data and never receive gradients.
- Real and synthetic targets remain distinguishable in metadata and logs.
- A counterfactual unit never trains a non-real endpoint ray.
- Exact Global receives real data only after warmup.
- Changing C always requires a C-conditioned model evaluation.

## 11. Current implementation gap

The current training code does not yet implement this full specification.
It currently supports one fake source paired with one real target and does not
construct the four-target counterfactual unit or the complete multi-C real
batch.

The next implementation should treat this document as the authoritative infra,
sampler, loss, and logging specification.

## 12. Pilot Stage E0: Endpoint-Only Counterfactual Training

This pilot is intentionally simpler than the complete Global-to-endpoint bridge
design. Its purpose is to determine whether counterfactual source correction can
strengthen endpoint control before introducing Global or any intermediate C.

### 12.1 Scope

Stage E0 has only the four existing environment endpoint contexts:

\[
C_A,\quad C_B,\quad C_C,\quad C_D
\]

It has:

- No Global context.
- No Global warmup.
- No interpolation between contexts.
- No near-Global or interior C sampling.
- No bridge-position weights.
- No Stage2 latent optimization.

Every training example is conditioned directly on the endpoint matching its
real environment.

The pilot starts from the successful four-endpoint Stage1 checkpoint. The four
endpoint context values remain frozen so that this experiment changes only the
model's ability to obey and recover the endpoint condition.

### 12.2 Training mixture

Each optimizer update uses:

\[
L_{\mathrm{E0}}
=
0.6L_{\mathrm{real}}
+
0.4L_{\mathrm{CF-endpoint}}
\]

The ratio is applied after independently normalizing the two branches.

One recommended macro-update contains:

- Three balanced real-data blocks.
- Two balanced endpoint-counterfactual blocks.
- One optimizer step after all five blocks have accumulated gradients.

Each block must be balanced over the four environments. Button/action coverage
must also be balanced so that a block does not identify an environment from an
irrelevant sampling artifact.

### 12.3 Real branch

For a real chunk from environment \(e\):

\[
z_e=\operatorname{VAE}(video_e)
\]

\[
x_\sigma=(1-\sigma)z_e+\sigma\epsilon
\]

\[
v^*_{\mathrm{real}}=\epsilon-z_e
\]

The model is evaluated directly at endpoint \(C_e\):

\[
L_{\mathrm{real}}
=
\left\|
v_\theta(x_\sigma,\sigma,a,C_e)
-
v^*_{\mathrm{real}}
\right\|^2
\]

This branch uses the normal full flow-matching timestep distribution. It is the
primary retention path for existing endpoint generation quality.

### 12.4 Smallest endpoint-counterfactual unit

Start with one real chunk from environment \(e\) and construct the same
initial-state, history, and action outcome under all four endpoint conditions:

\[
Z_b=
\left\{
z_b^{(A)},z_b^{(B)},z_b^{(C)},z_b^{(D)}
\right\}
\]

Exactly \(z_b^{(e)}\) is real. The other three videos are frozen-Teacher
counterfactual generations.

Select the source uniformly from all four versions:

\[
u\sim\operatorname{Uniform}(A,B,C,D)
\]

Therefore, inside the 40% counterfactual branch:

- 25% of source selections use the real source.
- 75% of source selections use a fake source.

Across the complete `60% real + 40% counterfactual` update, this corresponds to:

- 70% real-source forwards.
- 30% fake-source forwards.

The 40% ratio refers to the counterfactual training objective, not to the raw
fraction of fake-source inputs.

### 12.5 Endpoint correction objective

Create the noised source:

\[
x_\sigma=(1-\sigma)z_b^{(u)}+\sigma\epsilon
\]

Because the condition is exactly endpoint \(C_e\), the environment weights
collapse to:

\[
w_e=1,\qquad w_{k\neq e}=0
\]

Although four target videos exist, only the real endpoint target contributes:

\[
v^*_{\mathrm{CF-endpoint}}
=
\frac{x_\sigma-z_b^{(e)}}{\sigma}
\]

\[
L_{\mathrm{CF-endpoint}}
=
\left\|
v_\theta(x_\sigma,\sigma,a,C_e)
-
v^*_{\mathrm{CF-endpoint}}
\right\|^2
\]

When the uniformly selected source is the real version \(u=e\), this target
reduces to ordinary flow matching:

\[
\frac{x_\sigma-z_b^{(e)}}{\sigma}
=
\epsilon-z_b^{(e)}
\]

When \(u\neq e\), the model must remove the wrong endpoint outcome and move
toward the real endpoint outcome selected by \(C_e\).

### 12.6 Timestep policy

The real branch uses the full normal timestep distribution.

The endpoint-counterfactual branch initially uses the established high-noise
distribution:

- 20% noise fraction in `[0.90, 1.00]`.
- 60% noise fraction in `[0.70, 0.90]`.
- 20% noise fraction in `[0.55, 0.70]`.

Multiple timestep buckets should appear in every effective update. The same
timestep may be shared inside a balanced comparison block, while separate
blocks independently sample their timestep.

### 12.7 Offline Teacher bank

Teacher generation is an offline preparation step. For every selected real
chunk:

- Preserve the real video as the diagonal endpoint result.
- Generate the other three endpoint-conditioned videos.
- Record real environment, generated endpoint, action, episode, frame range,
  initial state, and Teacher checkpoint.
- Cache all four VAE latents.
- Keep real and synthetic provenance explicit.

Training must not run the Teacher or regenerate counterfactual videos inside the
optimizer loop.

The minimum valid bank entry contains one real video and exactly three Teacher
counterfactual videos for the same history and action.

### 12.8 Freeze and train policy

For the first pilot:

- Freeze all four endpoint context values.
- Do not instantiate a Global context.
- Train the same selected model modules across both branches.
- Do not use a separate editing branch or source-token branch.
- Do not apply lamp-region or object-region loss weighting.
- Keep the Teacher fully frozen and detached.

Using the same trainable modules in both branches ensures that the real replay
directly constrains any changes introduced by counterfactual correction.

### 12.9 Required metrics

Log at least:

- Normalized real loss.
- Normalized endpoint-counterfactual loss.
- Combined `0.6/0.4` loss.
- Per-environment real loss.
- Per-environment counterfactual loss.
- Real-source counterfactual loss.
- Aggregate fake-source counterfactual loss.
- Per-source-endpoint correction loss.
- Timestep and noise-fraction band.
- Endpoint real-rollout retention metrics.
- Counterfactual correction rollout metrics.

The logs must make the `40% counterfactual branch` and the resulting `30%
fake-source forwards` distinguishable.

### 12.10 Evaluation

For every endpoint environment, evaluate:

- Standard real endpoint rollout from the original real input path.
- Correction when the source is the matching real version.
- Correction from each of the three wrong endpoint versions.
- Ground-truth, pre-pilot checkpoint, and post-pilot checkpoint videos.
- Flow loss under all four source choices.

The main questions are:

- Does endpoint conditioning override an incorrect source outcome?
- Does the 60% real branch preserve existing endpoint quality?
- Is correction strongest in the high-noise region as expected?
- Does the model improve for all four environments rather than only one button
  outcome?

### 12.11 Success and failure criteria

Stage E0 succeeds if:

- Fake-source inputs are corrected toward the real endpoint outcome.
- Standard endpoint rollouts remain comparable to the starting checkpoint.
- Improvement is balanced across all four environments.
- The model uses endpoint C rather than simply preserving visible source state.

Stage E0 fails if:

- Real endpoint rollout quality degrades substantially.
- The model preserves the fake source outcome despite endpoint conditioning.
- Correction works only at almost-complete noise.
- One environment improves while another endpoint collapses.
- Loss improvement is caused only by texture matching without causal lamp-state
  correction.

### 12.12 Implementation phases

Phase E0-A prepares and validates the complete four-version Teacher bank.

Phase E0-B adds a default-off endpoint-counterfactual sampler and loss path. It
must not alter legacy Stage1, bridge, curriculum, or inference behavior.

Phase E0-C performs a self-terminating low-priority smoke test covering one real
block and one counterfactual block for every environment.

Phase E0-D runs a short trend experiment before any full training allocation.

No full Stage E0 training should be submitted until the Teacher bank,
source-provenance logs, real-retention evaluation, and smoke test all pass.
