# Paper Open Decisions

Decisions deferred for the AAAI-26 submission (full paper due 2026-07-28).
Each item lists the choice space, the trade-offs, and the current
recommendation. Decisions update this file in place; the implementation
follows the chosen option only after this file records it as resolved.

Cross-references:
- [`truck_pdm_scorer_design.md`](./truck_pdm_scorer_design.md) -- scorer architecture
- `project_map_free_pdms` memory -- sub-metric keep/rewrite/drop rationale
- `project_baseline_vs_kinematic_head` memory -- trailer-head ablation axes
- `project_paper_deadline` memory -- AAAI-26 schedule


## D1. Paper framing -- what is the headline contribution?

**Status:** open

The downstream choices (D2 trailer-head, D3 metric design weights, D4
ablation axes, D5 result table shape) all hinge on this.

| Framing | Headline | Body |
|---|---|---|
| **α: Metric-first** | "Articulation-aware PDMS metric for truck-trailer driving evaluation" | Main result = metric design + qualitative failure cases vanilla PDMS misses. Baselines (TruckTransfuser, DiffusionDrive) demonstrate the metric on different model families. Trailer head is method nuance, not a primary axis. |
| **β: Model-first** | "Trailer-aware planning + articulation-aware evaluation" | Main result = trailer-head models > tractor-only on our metric. Metric is the evaluation backbone, ablation table shows model architecture matters. |

**Trade-off:**
- α is cheaper in compute (no trailer-head retraining required) but
  thinner in model-side contribution.
- β requires the 2x2 trailer-head ablation (both TruckTransfuser AND
  DiffusionDrive get trailer heads -- ~3 extra days of code +
  ~25h GPU) but tells a richer story.

**Recommendation:** α. Hooks paper hook to metric (the actual code
contribution we are spending time on). Trailer-head studies become
follow-up.

**Open until:** D2 below settles.


## D2. Trailer-head ablation strategy

**Status:** open

Tied to D1. Four sub-options:

| Option | Setup | Cost | Paper symmetry |
|---|---|---|---|
| **A**: trailer head on β only | TruckTransfuser trained with and without trailer head; DiffusionDrive only tractor-only | 1 day + ~10h GPU | Asymmetric -- reviewer will ask "why only β?" |
| **B**: trailer head on β AND γ | Add trailer-head branch to V2TransfuserModel; retrain γ both variants | 1-2 days code + ~25h GPU | Symmetric 2x2 |
| **C**: no trained trailer head; kinematic propagation derives trailer trajectory at eval time | Both baselines stay tractor-only. A `propagate_trailer(tractor_traj, hitch_angle_0, trailer_L)` bicycle-model function turns the predicted tractor SE(2) into a trailer SE(2) post-hoc, fed into articulation PDMS. | 1 day (propagator) | Symmetric; paper line is "tractor-only baselines, articulation metric still applies" |
| **D**: drop trailer head from paper entirely | No mention of trailer head in main text | 0 | Cleanest but loses one axis |

**Trade-off:**
- A is fastest but reviewer-vulnerable.
- B is most complete result table; cost is real but inside the
  deadline budget.
- C reframes trailer dynamics as evaluation-time assumption rather
  than learned model output. Becomes a small method contribution
  ("kinematic propagator + articulation PDMS").
- D is purest metric-paper.

**Recommendation under D1=α:** C. Adds a small method-side contribution
(kinematic propagator), keeps result table symmetric, no extra
training.

**Recommendation under D1=β:** B. Required for the 2x2 ablation that
β's framing depends on.

**Pending:** awaits D1.


## D3. Articulation metric design parameters

**Status:** open (placeholder values in design doc)

Listed in `truck_pdm_scorer_design.md` § 7. Each is a paper-stage
tuning decision that will be set after Phase 1 scorer is up and we
inspect the score distributions on TruckScenes:

| Item | Placeholder | Decision criterion |
|---|---|---|
| `D_max` for off-tracking penalty | 1.5 m | Inspect distribution of log off-tracking values across TruckScenes train. Pick threshold that makes ~5-10% of log frames cross zero score (heavy turn maneuvers). |
| `W_corridor` for log_corridor compliance | 4-5 m | Articulated truck lateral footprint ~2.5 m. Corridor must be wider than that; default to 2x footprint = 5 m. |
| log_corridor classification: SafetyMetric (mult) vs QualityMetric (additive weight) | safety | If too noisy / many false positives, downgrade to quality with weight ~3. |
| Hitch-angle stability metric inclusion | optional | Ship only if Phase 2 budget remains. Defines a paper sub-result on jackknife prevention. |
| Traffic-light compliance fall-back | check devkit | If TruckScenes has no traffic-light annotation, return 1.0 (treat as no violation possible); otherwise reuse vanilla observation-driven logic. |

**Recommendation:** Defer all five until Phase 1 scorer runs against
real data; set values from log-distribution evidence. None block paper
table draft.


## D4. Result table axes

**Status:** open

The paper's main result table is the cartesian product of:

```
{baselines}  x  {metrics}
```

Open dimensions:

| Axis | Options | Notes |
|---|---|---|
| Baselines | (a) TruckTransfuser + TruckDiffusionDrive only | Two distinct architectures -- enough for paper baseline comparison |
| | (b) + Constant-velocity + ego_status_MLP from navsim | Free additions from navsim's vanilla agents; cheap to evaluate, makes table thicker |
| Metrics | (a) Phase 1 (no-map navsim PDMS) only | Demonstrates infrastructure works |
| | (b) Phase 1 + Phase 2 articulation extensions | Demonstrates paper contribution |
| | (c) + vanilla PDMS (where applicable) for contrast | "What vanilla PDMS misses" qualitative cases |
| Splits | (a) val only | matches navsim convention |
| | (b) val + curvy-only subset | TruckScenes is highway-heavy; turns are rare but they are exactly where articulation matters. A "curvy subset" carved from val makes articulation metric differences stand out. |

**Recommendation:**
- Baselines: (b) -- include trivial baselines as table rows for context (free).
- Metrics: (b) -- the paper contribution table without Phase 2 would
  be hollow.
- Splits: (b) -- argue that articulation metric is *measurable*
  across the full val, but the *signal* is concentrated in turns.


## D5. DiffusionDrive ego status input -- shortcut-free protocol

**Status:** resolved (2026-05-16)

**Decision:** option (B) from earlier discussion -- fork DiffusionDrive's
status encoder to use 3D driving_command only, no ego_velocity /
ego_acceleration. Aligns with TruckTransfuser's VAD §4.2 protocol
(`use_ego_status=False`).

**Rationale:** vanilla DiffusionDrive uses 8-D status (cmd 4D + vel 2D
+ acc 2D), known to inflate open-loop L2 by ~2x via vx*dt shortcut.
For paper fairness, both baselines must run the same shortcut-free
input protocol.

**Implementation:** committed (`3ecdb1d` -- 3D status protocol; `2072d2f`
-- bev_semantic gating).


## D6. Trailer ground truth definition

**Status:** half-resolved

TruckTransfuser uses *hitch-corrected* trailer center (5th-wheel
anchored, MAN TGX viewer convention). See
`navsim/common/truckscenes/trailer_extras.py`.

**Open sub-question:** does articulation PDMS evaluation use the same
hitch-corrected center, or the raw `vehicle.ego_trailer` annotation
center?

**Recommendation:** hitch-corrected (consistency with training-time
GT). Document in method section.

**Pending:** confirm raw vs corrected gives different paper numbers
when Phase 1 scorer runs.


## D7. Open-loop vs closed-loop evaluation

**Status:** mostly resolved

Navsim v1.1 evaluation is open-loop (predicted future trajectory
scored against log GT). v2 added two-stage reactive sim -- we are on
v1.1 by design (`project_dual_repo_structure` memory).

**Decision:** open-loop only for the paper.

**Rationale:** v2 closed-loop requires HD map for reactive agent
policies; TruckScenes has none. Open-loop is sufficient to validate
the metric.

**Future work bullet:** "closed-loop simulation extension when
TruckScenes HD map becomes available."


## D8. Baseline checkpoint provenance

**Status:** open

Two distinct training paths produced TruckTransfuser checkpoints:

| Path | Checkpoints | Status |
|---|---|---|
| transfuser-truckscenes/train.py (custom loop) | v9_heading_seed0, v9_lateral_seed0 (both 48 ep complete) | Already trained 2026-05-14 |
| navsim-truckscenes Lightning trainer (β) | truck_navsim_v9_seed0 (in progress, ~5-7h) | Same protocol, navsim form |

**Question:** which checkpoint do paper tables cite?

Options:
| Option | Pro | Con |
|---|---|---|
| Navsim-form (β) | Code release point matches; one trainer | Less battle-tested |
| transfuser-truckscenes-form (v9) | Already-validated; prior loss curves on hand | Different trainer than the one we release |
| Both, with parity check | Strongest paper claim | Extra evaluation pass |

**Recommendation:** β -- the navsim-form checkpoint is what is in the
released codebase. Cross-reference v9_heading numbers in supplementary
to show consistency.


## D9. Are we including a constant-velocity sanity baseline?

**Status:** open (related to D4)

navsim ships a `ConstantVelocityAgent`. Free to evaluate (no training);
its absurdly-bad PDMS confirms the metric discriminates between
learned and trivial policies.

**Recommendation:** include. Cheap, adds credibility, classic baseline
sanity.


## D10. driving_command derivation mode (heading vs lateral)

**Status:** open -- ablation runs queued (2026-05-16)

Current default in `scene_dict_from_sample.py` is heading-mode (|Δyaw@4s|
threshold 15°), mirroring nuPlan/navsim's route-derived intent
convention. The lateral-mode alternative (|local_y@4s| threshold 2m,
VAD's nuScenes-converter pattern) catches lane changes that heading
misses.

| Mode | Catches | Misses | Convention |
|---|---|---|---|
| heading (15°) | intersection turns | lane changes | nuPlan / navsim (route-derived) |
| lateral (2m) | turns + lane changes | gradual curves (big Δyaw, small Δy) | VAD / nuScenes converter |

**Argument for lateral:** TruckScenes has no HD map / route plan, so the
nuPlan route-derived semantics do not apply directly. Trajectory-derived
signal is what we have, and lateral captures more semantic intent
(lane changes are real maneuvers that should be observable to the
model). Highway truck dataset has many lane changes; heading misses
all of them.

**Argument for heading:** transfuser-truckscenes v9_cmd_no_status
(already-trained baseline) uses heading. Keeping heading allows 1:1
comparison with that prior checkpoint set without re-deriving.

**Decision plan:**
- Train both variants on the navsim form (`truck_navsim_v9_seed0` =
  heading [running], `truck_navsim_v9_lateral_seed0` [queued]).
- After Phase 1 scorer runs, compare PDMS on both variants.
- Pick the better-performing one as the paper's main result; cite the
  other as an ablation row.

**Implementation note:** `scene_dict_from_sample.py` now exposes a
`driving_command_mode` parameter; `build_navsim_logs.py` exposes a
matching CLI flag. Lateral pkls live in a separate output dir so the
two pkl families do not collide.


## D11. BEV range -- asymmetric (v9) vs symmetric wider (BEV-48)

**Status:** open -- ablation run queued (2026-05-16)

v9_cmd_no_status uses `x: -32 / +48, y: ±32` (forward-asymmetric 80m
× 64m). The intuition is "highway truck sees far forward, narrow
lateral." But this matters: if our articulation metrics emphasize
turning maneuvers, a wider lateral BEV may help.

**Candidate:** `x: ±48, y: ±48` (symmetric 96m × 96m). Pros:
- Wider lateral coverage -> better at perceiving adjacent lanes /
  turning agents
- Symmetric -> easier to reason about in paper / figures
- B200 has plenty of memory budget

Cons:
- 2.25x more BEV pixels -> slower forward (~25% wall clock per epoch)
- Backward range 48m is wasteful for truck-forward driving

**Decision plan:** train `truck_navsim_v9_bev48_seed0` (heading mode,
BEV ±48), compare to v9 baseline (current). Pick the better.


## D12. Trailer-load stratification -- "empty trailer" is yard, not road

**Status:** resolved (2026-05-19, based on `tools/checks/check_dynamics_load_split.py`
and visual inspection of all 22 empty-trailer scenes via
`ts.render_sample`)

The `scene_container_split.json` (built off median ego_trailer height
≥ 2.5 m) divides the 464 with-trailer scenes into:

  - **loaded** 442 scenes (median height ≥ 2.5 m -> container on
    chassis), from many source recordings, normal road driving.
  - **empty** 22 scenes (median height < 2.5 m -> chassis only).

We originally planned to group `no_trailer` + `empty_trailer` as a
"low-articulation" bin against `loaded_trailer` as "high-articulation"
to study the trailer-load effect. But once we rendered every empty
scene and ran a GT dynamics comparison, we found:

**Empty trailer scenes are container-terminal / yard operations, not
road driving.** All 22 originate from only **3 source recordings**
(`scene-c1b21af2... -> 14 sub-scenes`, `scene-0044384a... -> 5`,
`scene-b8e7da1b... -> 3`) — all of which depict the same kind of
context: a tractor with bare-chassis trailer maneuvering around stacked
shipping containers, gantry cranes, reach stackers, and yard workers.

This shows up in the GT dynamics medians vs loaded scenes:

| metric (median) | empty (yard) | loaded (road) | reading |
|---|---|---|---|
| \|hitch\| (deg) | 0.98 | 0.39 | yard truck parks at angle -> non-zero hitch |
| \|dhitch/dt\| (deg/s) | **0.02** | 0.32 | yard truck idle -> rate ≈ 0 |
| \|ay\| (m/s²) | 0.67 | 0.66 | only metric not confounded |
| \|yaw rate\| (deg/s) | **0.00** | 0.30 | yard idle vs highway turning |

i.e. three of the four "dynamics" differences are driven entirely by
the **driving context** (yard vs road), not by the trailer load. The
TruckScenes dataset has no road-driving samples of an empty trailer.

For comparison, the 134 `no_trailer` scenes span 12 source recordings
and are uniformly Autobahn-style highway footage (verified on 12
random middle-sample renders).

**Implications for the paper protocol:**

  1. **"Trailer-load effect" cannot be isolated** with this dataset.
     The empty bucket is confounded by being in a different operational
     domain. Any L2 or PDMS gap between empty and loaded is mostly the
     yard/road domain gap.

  2. The "trailer-presence" axis (no_trailer vs loaded_trailer) IS
     measurable -- both buckets are road driving -- and is what the
     paper articulation comparison should use:
       - **no_trailer**: tractor-only highway driving
       - **loaded_trailer**: tractor + container highway driving

  3. **Empty trailer is sanity-check / OOD only**, not a main result
     axis. Possible appendix study: "Does the model handle yard
     operations (OOD)?" Treat as out-of-distribution evaluation, not
     as a low-articulation in-distribution comparison.

  4. The original `low + high = no_trailer + empty vs loaded`
     grouping in D2/D4 needs to be retired in favour of:
        low-articulation = no_trailer
        high-articulation = loaded_trailer

**Counts (train / val) after this redefinition:**

| group | train scenes | val scenes |
|---|---|---|
| no_trailer | 116 | 18 |
| loaded_trailer | 388 | 54 |
| empty (yard, excluded from main metric) | 19 | 3 |

Cross-references:
- Analysis script: `transfuser-truckscenes/tools/checks/check_dynamics_load_split.py`
- Visual evidence: `transfuser-truckscenes/data/analysis/dynamics_load_split/empty_scenes/*.png` (all 22 rendered) and `.../no_trailer_scenes/*.png` (12 random renders).


Active decision queue (chronological)

| Order | Decision | Blocks |
|---|---|---|
| 1 | D1 framing (α vs β) | D2 |
| 2 | D2 trailer-head strategy | implementation of trailer-head retraining (if any) |
| 3 | D3 metric parameters | Phase 2 metric finalization (set after Phase 1 runs) |
| 4 | D4 result table axes | Evaluation script layout |
| 5 | D6 trailer GT consistency | (low priority, decided at eval time) |
| 6 | D8 baseline checkpoint provenance | Final table values |
| 7 | D9 constant-velocity inclusion | (cheap, last-mile decision) |

Resolved already: D5 (shortcut-free protocol), D7 (open-loop only).
