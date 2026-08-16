# Duck Harness — Implementation Plan

Retrofit the Tufa Labs duck-harness ARC-AGI-3 agent with a world-model
control loop, plus supporting efficiency fixes.

Target repo: fork of `Tufalabs/duck-harness`, work under
`src/ARC3-Inference/inference/`.

---

## 0. Read this first — the constraint that shapes every decision

Scoring is RHAE (Relative Human Action Efficiency):

```
level_score = min(1.15, (human_actions / agent_actions) ** 2)
game_score  = sum over solved levels of (level * level_score)
              / sum over all levels of level
```

Two consequences:

1. **The square punishes exploration.** Every hypothesis the agent tests
   costs a real environment action. The current agent learns by poking the
   world, which is exactly what the metric penalises.
2. **Efficiency is cheaper than capability.** For a game where the agent
   clears 3 of 8 levels, moving from 3x to 1.5x human actions takes the
   game score from 1.9% to 7.4% — a 4x gain with zero new levels solved.

Priority order follows directly: efficiency on already-solved games >
depth on partially-solved games > breadth on zero-scoring games.

**Level weighting matters for where to spend actions.** Weights are
`w_l = l`, so exploration on level 1 is nearly free, while a certified
world model pays off on the heavily-weighted deep levels. The model
transfers across levels within a single pass.

---

## 1. Design constraints — do not violate these

- **Within a pass, across levels: full carry-forward.** Levels are
  sequential inside one `HarnessSolver`. The world model, notes, and
  transition log persist from level 1 to level 9. This is the main thing
  the retrofit exploits.
- **Across passes: zero carry-forward, by design.** Each pass is an
  independent attempt with a fresh solver. Do NOT persist `world_model.py`
  to disk across passes to warm-start later attempts. The human baseline is
  first-time play; cross-pass memory breaks the premise the metric rests on
  and will not generalise to the private set.
- **Game-agnostic tooling only.** Tufa found hand-crafted tools hurt — but
  what hurt was *game-specific* tooling. Backtest and BFS encode no prior
  about what the games are, so they are consistent with the thin-harness
  philosophy.
- **Enforcement, not permission.** Where the plan says the solver must
  require a workflow, it means require. A merely-permitted workflow gets
  skipped by the model under pressure. This distinction is the single
  largest reported effect in the whole plan.

---

## 2. Phase 1 — Persistent world model + backtest certification

### Problem

`python_tool_sandbox.py` is ephemeral per call. Nothing survives context
eviction, so the agent re-derives its theory of the game every turn and pays
environment actions to do it.

### Change

**2.1 Persist agent working memory.**

Make `world_model.py` and `notes.md` persist across tool calls within a
single game run and survive context eviction. This persistence *is* the
agent's working memory.

- Files live in the run's job dir, scoped per game run
- Must survive the oldest-message eviction that bounds context
- Must NOT persist across passes (see Section 1)

**2.2 Add `run_backtest()` tool.**

Replays the induced `step(state, action)` over the full recorded transition
log.

Returns:
- `N/M exact` — how many recorded transitions the model reproduces exactly
- first mismatching frame diff — so the agent knows precisely what it got wrong

The transition log already exists in `*_events.jsonl`. This is the highest
value-per-effort addition in the plan.

### Why it works

Building and checking a theory costs zero environment actions. Only
gathering new evidence costs actions. The agent can rewrite and re-check a
hundred times for free.

### Reference result

The Schema harness reports 98.98% on the public 25 with frontier models
versus 42.83% for the same models under a generic Claude Code harness —
+56pp from process alone. In the 14/25 games where an exactly-reproducing
program was induced, the agent used 1.6–5.0x *fewer* actions than humans.

### Worked example of the loop

| Action | Observation | Inference |
|---|---|---|
| UP | 3x3 cyan blob shifts up one cell, nothing else changes | that blob is `player` |
| UP | blob moves again | UP implies `player.y -= 1` |
| UP | nothing changes at all | grey cells above are `wall` |
| LEFT | blob moves, a red cell vanishes, bottom bar grows | red = collectible, bar = counter |

Four actions, three object types and two rules invented — none of which
were given. Write as `step()`, backtest against all four transitions, then
plan routes through walls never touched, because the rule generalises.

### Acceptance

- `world_model.py` contents survive a forced context eviction mid-game
- `run_backtest()` on a deliberately wrong `step()` returns the correct
  first-mismatch index
- No cross-pass leakage: two passes of the same game start with empty state

---

## 3. Phase 2 — Search inside the certified model

### Problem

Even with correct mechanics, the agent finds paths by walking them. Every
wrong turn is a squared penalty.

### Change

**3.1 Add `run_bfs()` tool.**

BFS / A* over the certified `step()` with an inferred `is_goal()`. Thousands
of states expanded for zero RHAE cost.

**3.2 Goal inference.**

`is_goal()` is learned the same way as the rules. The environment reports
`level_completed`; the agent inspects the preceding state, writes a guess
(e.g. "all red cells gone"), and backtests that guess against the log like
any other hypothesis.

**3.3 Certification gate.**

Planning is only permitted once `run_backtest()` reports all recorded
transitions reproduced exactly. Do not let the agent search inside an
uncertified model.

### Reference result

On one Schema level a human needed 500 actions; BFS-in-model needed 42.

### Acceptance

- BFS refuses to run against a model with any backtest mismatch
- Solved-level action counts drop measurably versus baseline on games the
  agent already clears

---

## 4. Phase 3 — Gated commit with abort-on-misprediction

### Problem

A model that backtests perfectly can still be wrong about something never
encountered.

### Change

Route all actions through a single gated channel: `commit_actions(queue)`.

- Each executed step is compared against the model's prediction
- The **first** mismatch discards the entire remaining plan
- Control returns to theorizing with the new contradicting transition added
  to the log

Modify `solver.py` so action submission cannot bypass this channel.

### Why enforcement matters

This must be enforced by the solver, not offered as an option. The 43% to
99% difference in the Schema result came from the harness *requiring* the
workflow rather than permitting it.

### Acceptance

- No code path can reach `GameAPI` without passing the prediction check
- An injected wrong model aborts on the first mispredicted step, not later

---

## 5. Phase 4 — Representation revision, not just rule revision

### Problem

Sometimes no rule fits, no matter how the agent patches it. The objects are
wrong, not the transition function.

### Change

When backtest failures persist across multiple rewrite attempts, the agent
must be prompted to revise the **state representation** — what the objects
are, and whether hidden state variables exist — rather than continuing to
patch the transition rules.

### Worked example

Schema's agent spent three levels treating a block as directly steerable.
Predictions kept failing. It eventually recorded that it had the block model
completely wrong and replaced it: the block moves when the **player's carry
state** changes. That required inventing a `carrying` variable that appears
nowhere in the grid. Mispredictions collapsed from dozens per level to four,
and levels 5, 8 and 9 fell at roughly a quarter of human actions.

### Note

Schema's Observation 2: Fable outperformed Opus not by knowing more
mechanisms, but by questioning the state representation earlier and choosing
more discriminating experiments. This is a learnable policy over
deliberation — see Phase 5.

---

## 6. Cheap wins — implement alongside Phase 1

| Fix | Reason | Mechanism |
|---|---|---|
| No-op cache | every repeated no-op is pure score destruction | `last_action_result.board_changed` already exists; hash `(state, action)` to no-op, never repeat |
| Oscillation guard | LEFT/RIGHT/LEFT loops are a classic LLM failure, burning actions quadratically | detect and break the cycle |
| Click-target pruning | 64x64 = 4096 MOUSE targets is an unusable search space | restrict to segmentation-derived candidates: component centroids, rare colours, small button-like shapes |
| Dead-signature memory | (Reki's technique) | if clicking an object *type* never changes anything, stop clicking that type for the rest of the level |
| Predict-before-act | bootstraps the world model and suppresses flailing at once | require a written predicted next frame before any action, then diff |
| Macro-action commit | also buys wall-clock headroom at high concurrency | once mechanics are known, submit 20–60 actions per turn |

---

## 7. Phase 5 — Trace distillation

### Problem

Tufa's own finding: solvability depends on model capability, cost depends on
the harness. Kaggle caps you at a 27B model on one GPU.

### Change

The pipeline already exists — `make traces` exports episodes in live-chat
`messages` format.

1. Run a frontier model through the harness on public games
2. Filter to high-RHAE episodes
3. SFT Qwen on those traces

### What to target

The representation-questioning behaviour from Phase 4 specifically. That is
what separates the strong models and what the traces capture.

---

## 8. Measurement discipline — read before claiming any improvement

**Baseline: 1.6002 ± 0.4475 on the public set.** The standard deviation is
28% of the mean. Most improvements measured at fewer than 20 passes are
noise.

Rules:

- Keep `make significance` — paired by game, require `P(delta > 0) >= 0.90`
- Treat anything below ~2x the paired SE as unproven
- Use `SIMULATE_COMPETITION_ARCADE=true` with `COMPETITION_CLONE_RUNS` for
  submission robustness
- **Hold out games.** Do not tune on all 25.

Two warnings:

- The milestone-1 third-place finisher reported that local public-game
  checks were not a reliable leaderboard proxy.
- Schema is explicit that a near-ceiling public score does not extrapolate
  to semi-private.

Iterate locally, not on Kaggle. The repo ships `SIMULATE_COMPETITION_ARCADE`
for exactly this, and Kaggle's ~30 GPU-hours/week only affords roughly 15–20
full validation runs.

---

## 9. Sequencing

| Weeks | Work |
|---|---|
| 1–2 | Phase 1 + Section 6 cheap wins. Measure at >= 20 passes. |
| 3–4 | Phase 2 + Phase 3. The efficiency step-change should appear here. |
| 5–6 | Phase 5 trace distillation. |
| 7 | Robustness only: wall-clock guards, graceful degradation if the vLLM server dies, full arcade simulation. |

A crash on the private set is a total loss. In the final week that risk
outweighs any marginal feature.

---

## 10. Deployment notes

The Kaggle notebook is infrastructure only — the agent is frozen in
`benchmark_initial.pkl` inside an attached dataset.

- Pickles store instance state, not class code. Unpickling re-imports
  `HarnessSolver` from whatever source is on `sys.path`, so rewritten method
  bodies and new tools are picked up automatically.
- **But**: adding an attribute that `__init__` now sets will `AttributeError`
  against an old pickled `__dict__`. Rebuild the bundle rather than swapping
  source files.
- Update `DATASET_SOURCES` in the notebook's dataset-resolution cell to your
  own slug.
- The post-unpickle customization cell is the sanctioned monkey-patch point
  for quick experiments without a full rebuild.
- Milestone eligibility requires open-sourcing under a permissive licence,
  with the Duck's source attributable as third-party work.

---

## 11. Aside — the paper route

The Paper Prize is a separate pool requiring only a *linked* submission, not
a high-scoring one. If the world-model retrofit produces a clean controlled
ablation against the Duck baseline, that is a paper-shaped result
independent of leaderboard position.