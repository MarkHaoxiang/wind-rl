# Geometric / Equivariant Architectures for wind-rl

**Date:** 2026-07-19 · **Scope:** architecture selection for (a) the generative layout
prior (few-step flow-matching over N≤92 turbine positions in 2D) and (b) the MAPPO
policy/critic over turbines. Successor to DiCoDe (arXiv:2511.03100). Decision-oriented.

**TL;DR.** Build a torch-native architecture-benchmark suite first, then climb a 3-rung
ladder: **v0** dense-adjacency GCN → **v1** permutation-equivariant set transformer with
relative-position features + rotation augmentation → **v2** SO(2)/circular-harmonic
("2D Equiformer") attention. The owner's hunch is correct and stronger than they realise:
in 2D the whole spherical-harmonic + Clebsch-Gordan apparatus **collapses to Fourier
modes on the circle**, tensor products become 1D circular convolutions (FFT-cheap), and
the eSCN "SO(2) trick" you'd apply to 3D is simply the *native* structure of 2D — no
Wigner matrices, no compiled extensions, plain torch. But equivariance is not obviously
worth it at N≤92 with cheap 1-parameter rotation augmentation, so v2 must earn promotion
on the benchmark, not by assumption.

---

## 1. Landscape

Notation: "type-ℓ / type-m features" = features transforming in irrep ℓ (3D) or m (2D).
Cost given per layer; E = edges, N = nodes, C = channels, L = max degree/frequency.

| Family | Core idea | Cost | Impl. weight | Library |
|---|---|---|---|---|
| **EGNN** (2102.09844) | Only scalars + a single type-1 coordinate vector; messages are MLPs of *invariant* pairwise distances, positions updated by relative-vector sums. No irreps, no CG. E(n)-equivariant, trivially cheap. | O(E·C) | Very light (~100 lines torch) | none needed |
| **PaiNN** (2102.03150) | Scalar + vector (ℓ=0,1) channels; equivariant message passing via gated products of scalar features and vector features. A "vectors-only" middle ground: more expressive than EGNN, no higher irreps. | O(E·C) | Light | none needed |
| **TFN / SE(3)-Transformer** (1802.08219 / 2006.10503) | Full steerable feature space up to degree L; messages built from spherical harmonics of edge directions contracted with features via Clebsch-Gordan tensor products; SE(3)-Transformer adds attention weights. Maximally expressive, canonical lineage. | O(E·L⁶·C) naive TP | Heavy (CG, `e3nn`) | e3nn |
| **Equiformer / EquiformerV2** (2206.11990 / 2306.12059) | Transformer with equivariant attention over steerable features; V2 swaps in the eSCN SO(2) convolutions + S² grid activations to scale to high L (up to 6-8). SOTA-ish on OC20/molecules. | O(E·L³·C) (V2) | Heavy | e3nn / fairchem |
| **eSCN** (2302.03655) | *The key primitive.* Rotate each edge so its direction aligns with a pole; this sparsifies the SO(3) tensor product into **band-diagonal SO(2) convolutions** (index m preserved), dropping cost O(L⁶)→O(L³). Equivalently: do the nonlinearity on an S² grid. | O(E·L³·C) | Medium-heavy | fairchem |
| **GotenNet** (ICLR 2025, OpenReview 5wxCQDtbMo) | High-degree *Cartesian* steerable features + geometry-aware tensor attention + hierarchical refinement, **without irreps or CG transforms**. Matches/beats EquiformerV2 on QM9/rMD17 at lower overhead. Strong "expressive but no CG" data point. | O(E·L·C) | Medium | reference repo |
| **Set Transformer** (1810.00825) | Permutation-equivariant/invariant attention over a set (ISAB/PMA inducing points → linear in N). *No geometry prior* — you feed coordinates as ordinary features. The natural v1. | O(N²·C) or O(N·M·C) | Light (torch native) | torch |
| **DiT** (2212.09748) | Plain transformer denoiser for diffusion/flow; adaLN conditioning. Permutation-equivariant if you drop positional encodings and treat points as tokens. The default modern generative backbone. | O(N²·C) | Light | torch |
| **EDM / equivariant flow-matching** (2203.17003; FM 2210.02747; SemlaFlow-style 2406.07266; SymDiff 2410.06262) | E(3)-equivariant denoiser (usually EGNN) inside diffusion/flow-matching over point sets; guarantees an invariant generated density. SymDiff shows you can get equivariance-in-distribution from a *non-equivariant* net via stochastic symmetrisation. | backbone-dependent | Medium | — |

**Cross-cutting 2024-26 finding for the debate:** "Do we need equivariant models for
molecule generation?" (2507.09753) and SymDiff (2410.06262) both argue a well-tuned
non-equivariant transformer + augmentation is competitive with equivariant models for
*generation* once data is sufficient — directly relevant to whether v2 beats v1.

---

## 2. The 2D specialisation (the owner's hunch — analysed)

**The math.** In 2D the rotation group is **SO(2)**, abelian. Its irreps are 1-D complex
characters indexed by integer frequency m: rotation by θ acts on a type-m feature as
multiplication by e^{imθ} (real form: a 2×2 block R(mθ)). m=0 is the invariant/scalar
irrep. The "spherical harmonics of 2D" are literally the **circular harmonics**
Y_m(φ)=e^{imφ}, i.e. Fourier modes on the circle. So the steerable edge embedding of a
direction φ is just the Fourier feature stack `[cos(mφ), sin(mφ)]_{m=0..L}` — O(L),
computed directly, no library.

**Tensor products = index addition = circular convolution.** Coupling type-m₁ ⊗ type-m₂
in the complex basis gives type-(m₁+m₂): c_{m₁+m₂} = a_{m₁}·b_{m₂}. The full contraction
`c_m = Σ_{m₁} a_{m₁} b_{m−m₁}` is the **Cauchy product / 1-D convolution** of the two
coefficient sequences. Since {a_m} are Fourier coefficients of an angular function
f_a(φ)=Σ a_m e^{imφ}, this convolution equals **pointwise multiplication of the two
angular functions**: f_c(φ)=f_a(φ)·f_b(φ). Hence the "FFT trick" the owner intuited:
iFFT the coefficient vectors to K sampled angles → multiply pointwise → FFT back, at
O(K log K) instead of O(L²) naive. Nonlinearities are applied pointwise **on the angular
grid** (the 2D analogue of EquiformerV2's S² grid activations), which is where all the
network's nonlinearity legally lives.

**Relation to eSCN.** eSCN's entire contribution is *manufacturing* this structure in 3D:
it rotates edges to a pole so the SO(3) tensor product decouples into per-order SO(2)
convolutions (m preserved, band-diagonal). **In 2D you start there** — no alignment
rotation, no Wigner-D matrices, no Clebsch-Gordan tables. The eSCN speedup is free and
exact. An "SO(2)-Equiformer" is therefore: (i) SO(2) linear layers = per-frequency
complex 1×1 maps (a batched matmul / grouped conv over the m axis; m=0 channels may mix
freely, m≠0 channels mix within |m|); (ii) tensor-product interaction via the FFT-grid
product above; (iii) attention with **invariant** logits (built from m=0 channels and
|·|² of type-m channels / inner products) modulating **equivariant** type-m values — the
Equiformer recipe, but with the CG machinery replaced by FFTs.

**Assessment of the three questions:**

- **(i) Novel-ish?** Yes, moderately. 2D steerable *CNNs on grids* exist (Harmonic
  Networks 1612.04642; general E(2)-CNNs / `escnn` 1911.08251), and the SO(2) machinery
  is textbook, but a **circular-harmonic irrep *attention transformer over point sets in
  the plane*** is under-explored as a named architecture — a legitimate small
  contribution for this domain, not a reimplementation of something off-the-shelf.
- **(ii) Plain torch, no compiled extensions?** Yes, cleanly. Complex tensors or cos/sin
  reals; `torch.fft` for the grid product; batched matmul for SO(2)-linear; standard
  softmax attention. No `e3nn`, no `torch-scatter`, no CUDA kernels. This is a genuine
  advantage of the 2D collapse and the main reason it's attractive here.
- **(iii) Will it beat a strong augmented transformer at N≤92?** **Uncertain — do not
  assume yes.** SO(2) is a *1-parameter* group; rotation augmentation covers its orbit
  almost perfectly and is nearly free, so the sample-efficiency edge of exact
  equivariance is smaller in 2D than in 3D. Recent evidence (2507.09753, 2410.06262)
  shows augmented non-equivariant transformers matching equivariant ones for generation
  at moderate data. Equivariance still buys: exact invariance guarantees (clean rotational
  coverage for the generator; consistency for the policy), better behaviour in the
  *low-data* early-RL regime, and a smaller model. **Verdict: build v2, gate it on the
  benchmark beating v1 by a real margin; if augmentation-v1 ties it, ship v1.**

**Wind-direction subtlety — the correct group action.**

- **Generator: E(2) = SO(2) ⋉ ℝ² (optionally O(2) with reflections).** Rotating the scene
  must co-rotate **positions, boundary-polygon vertices, *and the wind rose*** together;
  the layout distribution then transforms equivariantly. A wind rose is a *function on the
  circle*, so its natural conditioning representation is precisely its **circular-harmonic
  (Fourier) coefficients** — it drops straight into a type-m feature slot and co-rotates
  correctly by construction. This is where equivariance pays most: the rose **cannot** be
  canonicalised to one direction, so genuine SO(2) structure is doing real work. Translation:
  center on the polygon centroid. Reflection: wake physics is chirality-symmetric, so **O(2)
  (include reflections)** is arguably the correct group — free extra symmetry.

- **Policy: diagonal SO(2), and it is only correct if wind co-rotates.** Under a global
  rotation by θ: positions rotate, wind direction φ→φ+θ, and each yaw output (an
  orientation) → yaw+θ. Power/reward is invariant under this **joint** action, **not**
  under rotating positions with wind held fixed (that changes the reward — naive
  position-only equivariance is *wrong*). Two correct routes: **(a)** feed wind as a
  type-1 (m=1) geometric feature; a genuine SO(2)-equivariant net then emits yaw outputs
  (type-1/angle) that co-rotate automatically — the principled generalisation of the
  predecessor's hand-built "wind-relative edge features". **(b) Canonicalisation
  (recommended baseline):** since there is a *single global* wind direction, rotate the
  whole scene into the wind frame (+x = wind), run a plain permutation-equivariant
  transformer producing wind-relative yaws, rotate back. This yields *exact* SO(2)
  invariance with **zero** equivariant machinery and is likely to make full policy
  equivariance unnecessary. **Implication: equivariance is a bigger lever for the
  generator (conditioned on a full rose) than for the policy (single wind → canonicalise).**

---

## 3. Constraint handling (min-distance + boundary polygon)

Constraints are hard (min pairwise spacing; each point inside a site polygon) and, for the
generator, *coupled* (min-distance is pairwise). Options:

1. **Guidance-time projection (recommended default — this is DiCoDe's PUG).** Keep the
   network unconstrained; project samples onto the feasible set during flow/diffusion
   sampling (a few iterations of pairwise repulsion for min-distance + per-point projection
   onto the polygon). Architecture-agnostic → keeps the benchmark clean and comparable
   across v0/v1/v2. Composes cleanly with equivariance: projection onto the min-distance /
   polygon set is itself E(2)-equivariant, so symmetry is preserved end-to-end.
2. **Baked-in projection/output layers.** A final layer maps to a feasible parameterisation.
   Works for box constraints, awkward for coupled min-distance; can distort training
   dynamics. Avoid for spacing; fine for the polygon via a smooth interior map only if it
   helps.
3. **Constraint-aware attention bias.** Add a distance-dependent bias discouraging close
   pairs. *Soft only* — no feasibility guarantee; useful as a training-time prior on top of
   (1), not a replacement.

**Recommendation:** constraints stay **out of the architecture**. Generator → PUG-style
projection at sampling (optionally soft bias (3) as a prior). Policy → yaw actions are box
constraints (tanh squash); boundary/min-distance do not bind during an episode (positions
are frozen), so they never touch the policy net. This deliberately **decouples the
constraint question from the architecture question** so the benchmark measures architecture
alone.

---

## 4. Recommended staged path

An **independent architecture-benchmark suite** (Section 5) gates every promotion. All
stages torch-native, no `torch-scatter` (N≤92 ⇒ N²≤~8.5k, so **dense adjacency**
`Â X W` or `index_add` is trivially cheap; KNN via `torch.cdist` + `topk`).

| Stage | What | Effort | Risk | Promotion gate (must beat previous rung) |
|---|---|---|---|---|
| **v0 GCN** | Dense-adjacency GCN/GraphConv on KNN graph (policy) / on the point set (generator denoiser). Coordinates + wind as raw node/edge features. Not permutation- or rotation-*principled* but a fast floor. | ~1 day each net | Low. Mainly a plumbing + harness shakedown. | Establishes baseline numbers + a green, fast benchmark harness. Not a research result — a floor. |
| **v1 Set transformer** | Permutation-equivariant attention (Set Transformer / DiT-style tokens), relative-position + distance features, **rotation augmentation** (generator) / **wind-frame canonicalisation** (policy). No irreps. | ~3-5 days each | Low-med. This is the *expected production baseline* and a genuinely strong one. | Beat v0 on all primary metrics; hit target feasibility (gen) / power-capture (policy) at acceptable sample/step time. |
| **v2 SO(2)/circular-harmonic** | "2D Equiformer": circular-harmonic edge embeddings, SO(2)-linear (per-frequency complex matmul), FFT-grid tensor products + grid activations, invariant-logit attention over type-m values. Wind rose → circular-harmonic conditioning (gen); wind as type-1 feature (policy). Plain torch + `torch.fft`. | ~1.5-3 weeks (shared core across both nets) | Med. *Technical* risk low (no exotic deps); *payoff* risk real — may only tie augmented v1 at N≤92. | Beat v1 by a **meaningful margin** on sample efficiency / feasibility / NLL-proxy at matched wall-clock, **and** show ~0 equivariance error. If it only ties v1, keep v1 and bank v2 as a research artifact. |

**Primary gate metrics.** *Generator:* feasibility rate (pre-projection), distributional
match (2-Wasserstein / Sinkhorn-OT or MMD to reference set-distribution), coverage/recall,
NLL-proxy (FM likelihood via the probability-flow ODE), sample time (NFE + wall-clock),
equivariance error under co-rotated (positions+polygon+rose). *Policy:* power capture at
matched wind frames on 8-turbine FLORIS, sample efficiency (power vs PPO iterations),
wall-clock/step, parameter count.

---

## 5. Benchmark suite design sketch (minutes-to-hours, no full co-design runs)

Goal: rank architectures without RL co-design loops. Two harnesses, both fast.

**Generator harness — fit a known point-set distribution.**
- *Task:* sample layouts from a **procedural reference** with known feasibility structure —
  e.g. **Poisson-disk / blue-noise** points in a site polygon at the true min-distance,
  optionally modulated by a synthetic wind rose (denser cross-wind). Ground-truth density &
  feasibility are analytic. Train the flow-map to match; hundreds of steps, minutes on one GPU.
- *Metrics:* (1) **feasibility rate** pre-projection (min-dist + in-polygon); (2)
  **distributional distance** via a permutation-invariant set metric — Sinkhorn-OT /
  2-Wasserstein or Chamfer/MMD on the empirical point measures; (3) **coverage/recall**
  (mode coverage across rotational orbit & N); (4) **NLL-proxy** from the FM ODE; (5)
  **sample cost** (NFE, ms/sample); (6) **equivariance error** = distributional shift when
  the conditioning (polygon + rose) is co-rotated (≈0 for v2, measurable for v0/v1).
- *Sweeps:* N ∈ {8, 24, 48, 92}; 1-2 polygon shapes; isotropic vs peaked rose.

**Policy/critic harness — no full RL.**
- *(A) Supervised value regression (fastest).* On **logged rollouts** (or a grid of random
  layouts × yaws × wind), regress FLORIS power (and/or critic targets). Report R²/MSE,
  wall-clock/step, params. Purely supervised → minutes. Tests geometric inductive bias and
  the canonicalisation vs equivariance question directly.
- *(B) Frozen-env few-iteration PPO.* On **8-turbine FLORIS** (the wfcrl env already in-repo),
  freeze layouts, run K short PPO iterations; report **power capture vs greedy/zero-yaw
  baseline** and the sample-efficiency curve (power vs iteration). Hours, not days.
- *(C) Symmetry check.* Feed co-rotated (positions + wind) frames; measure power-prediction
  consistency and, for canonicalised models, confirm exact invariance.
- *Metrics:* power capture at matched frames, R²/MSE (regression), sample efficiency,
  wall-clock/step, params.

**Harness principles:** identical data/seeds/optimiser budget across architectures; report
**quality-vs-wall-clock Pareto**, not quality alone (a tie on quality at 3× cost is a loss);
promotion is a scripted threshold check, not eyeballing.

---

## 6. References (arXiv)

- **DiCoDe / this project's predecessor** — Scaling Multi-Agent Environment Co-Design with
  Diffusion Models. arXiv:2511.03100.
- **EGNN** — E(n) Equivariant Graph Neural Networks. arXiv:2102.09844.
- **PaiNN** — Equivariant message passing for tensorial properties. arXiv:2102.03150.
- **TFN** — Tensor Field Networks. arXiv:1802.08219.
- **SE(3)-Transformer** — arXiv:2006.10503.
- **Equiformer** — arXiv:2206.11990. **EquiformerV2** — arXiv:2306.12059.
- **eSCN** — Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs.
  arXiv:2302.03655. *(the SO(2) trick; the crux of §2)*
- **GotenNet** — Rethinking Efficient 3D Equivariant GNNs. ICLR 2025, OpenReview 5wxCQDtbMo.
- **Set Transformer** — arXiv:1810.00825.
- **DiT** — Scalable Diffusion Models with Transformers. arXiv:2212.09748.
- **EDM** — Equivariant Diffusion for Molecule Generation in 3D. arXiv:2203.17003.
- **Flow Matching** — Lipman et al. arXiv:2210.02747.
- **SemlaFlow / molecular flow-matching (scale-OT)** — arXiv:2406.07266.
- **SymDiff** — Equivariant Diffusion via Stochastic Symmetrisation. arXiv:2410.06262.
- **"Do we need equivariant models for molecule generation?"** — arXiv:2507.09753.
  *(equivariance-vs-augmentation evidence for the v2 gate)*
- **Harmonic Networks** — circular-harmonic 2D CNNs. arXiv:1612.04642.
- **General E(2)-Equivariant Steerable CNNs (`e2cnn`/`escnn`)** — arXiv:1911.08251.
- **e3nn** — arXiv:2207.09453 (3D reference; *not* needed for the 2D path).
