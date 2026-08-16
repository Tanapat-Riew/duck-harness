# Duck Harness — Improvement Plan v2 (consolidated)

Supersedes `improvment-v1.md`. Same thesis, corrected for what the codebase
actually permits and for the model you actually get on Kaggle.

Changes from v1 are marked **[revised]**. Everything unmarked is v1's reasoning,
kept because it holds up.

---

## 0. The constraint

Scoring is RHAE:

```
level_score = min(1.15, (human_actions / agent_actions) ** 2)
game_score  = sum over solved levels of (level * level_score)
              / sum over all levels of level
```

The square punishes exploration, and every hypothesis the current agent tests
costs a real environment action. `PYTHON_ADDENDUM` literally prescribes the
expensive loop: *"If confidence is low, program a discriminating probe and
revise the world model from the result."*

Efficiency is cheaper than capability: on a game where the agent clears 3 of 8
levels, going from 3x to 1.5x human actions moves the game score from 1.9% to
7.4% with zero new levels solved.

Priority order: efficiency on already-solved games > depth on partially-solved
games > breadth on zero-scoring games.

---

## 1. Design constraints

- **Within a pass, across levels: full carry-forward.** Levels are sequential
  inside one `HarnessSolver`. The induced model persists from level 1 to level 9.
- **Across passes: zero carry-forward.** Each pass is an independent attempt.
  The human baseline is first-time play.
  **[revised]** This needs no new enforcement. `ToolAgent._ensure_session`
  already clears per-game state when the runtime directory changes. Keeping the
  induced model **in memory on the agent instance** rather than on disk makes
  cross-pass isolation the default instead of something to police. v1's "files
  live in the run's job dir" would have to be actively defended against leakage.
- **Game-agnostic tooling only.** Backtest and BFS encode no prior about what
  the games are.
- **Enforcement, not permission.** A merely-permitted workflow gets skipped
  under pressure.
  **[revised]** With one exception — see the fallback in §3.

---

## 2. What changes, in one sentence

The agent stops paying environment actions to test hypotheses, because it can
test them for free against transitions it already recorded.

---

## 3. Phase 1 — Induced model + backtest certification

### The blocker v1 missed

v1 §2.1 says "make `world_model.py` and `notes.md` persist across tool calls."
Model code cannot do this. The sandbox has:

- no `open` — it is absent from `SAFE_BUILTINS`
- no `os`, no `pathlib` — `_safe_import` rejects everything outside a
  16-module allowlist, so `import world_model` raises `ImportError`
- a per-call `TemporaryDirectory` as cwd, deleted when the call returns

Widening that boundary is the wrong fix; it is what keeps generated code away
from your run artifacts.

### [revised] The design that works

Persistence is **host-mediated**. The model never touches a file.

| Piece | Where it lives | Contract |
| --- | --- | --- |
| `save_model(source)` | sandbox global | compiles `source`, requires callable `encode` and `step`, returns `{"ok", "error"}` |
| induced source | `ToolAgent._world_model_source` | in-memory, per game run, cleared by `_ensure_session` |
| `encode` / `step` / `is_goal` | sandbox globals | re-`exec`'d into the namespace at the start of every call |
| `run_backtest()` | sandbox global | replays `transitions`, returns `N/M exact` plus first mismatch |

The model writes three functions:

```python
def encode(frame):  ...   # frame -> hashable state
def step(state, action): ...   # state, action -> state
def is_goal(state): ...   # optional, Phase 2
```

### [revised] Why `encode` solves the deadlock v1 would have hit

v1's certification gate — "planning permitted only once backtest reproduces
every transition exactly" — assumes exact reproduction is attainable. Your own
prompts say otherwise: a single action can produce a multi-frame animation, and
`VISUAL_GAME_ADDENDUM` describes edge bars that change every step. A timer that
decrements each action means `step()` must model the timer exactly or every
transition mismatches forever.

Making the agent own `encode` fixes this at the right level: an `encode` that
drops the HUD produces a state space where the timer does not exist, so
certification is attainable without the harness guessing what to mask.

This is a better answer than harness-side frame masking, which was my first
suggestion — it is wrong because the harness does not know which regions matter,
and being wrong there silently caps the ceiling.

The harness still owes the agent the *evidence* to discover this: see §4.

### [revised] The fallback — the one place enforcement must yield

v1's certification gate converts "mediocre agent" into "agent that never acts"
if the model cannot induce a correct `step()`. Schema's 98.98% came from
frontier models; Kaggle gives you one 27B. Writing a BFS is easy for that model;
inducing and debugging a transition function for a novel game is not, and that
is the load-bearing skill in Phases 1–4.

So the gate needs a documented escape: after **K failed revision rounds or T
seconds without certification**, fall back to today's reactive policy and record
that it happened. The fallback rate is itself the measurement that decides
whether Phases 2–4 are worth building.

v1 puts robustness in week 7. This particular piece of robustness is week 1,
because without it a bad induction phase produces zero actions and a zero score.

---

## 4. Phase 1b — Give the agent evidence it cannot compute itself

The harness sees every `(action, before, after)` transition. The model only sees
`board_changed: bool`. Two things the harness should compute:

**Board-change report.** Per action: changed-cell count, bounding box, and
whether every changed cell sits within N cells of a border. This attacks the
failure mode `VISUAL_GAME_ADDENDUM` currently spends a caps-lock paragraph
begging about ("DON'T DO THIS!"), and it is the signal that tells the agent its
`encode` is picking up HUD state.

**No-effect memory.** Actions that produced no board change, remembered for the
level. The model cannot maintain this — it needs cross-turn host state — and
re-testing a dead action is pure squared-penalty waste.

Rule of thumb for what belongs in the harness versus the prompt: **the harness
should own what the model cannot compute, and stop nagging about what it can.**

---

## 5. Phase 2 — Search inside the certified model

BFS / A* over `step()` with an inferred `is_goal()`. Thousands of states expanded
for zero RHAE cost. Goal inference is learned like any other rule: the
environment reports `level_completed`, the agent guesses ("all red cells gone")
and backtests the guess.

**[revised] Budget check.** `_python_timeout` is `min(30, ...)` — a hard 30s per
call — inside a 60s turn yield, in a subprocess with `RLIMIT_CPU` and 32 file
descriptors. Backtest is cheap and belongs in the sandbox. A BFS over thousands
of states may not fit, and `run_bfs` may need to be a host-side tool with its own
budget. Decide this with a measurement, not in advance.

**Gated on certification**, with the §3 fallback as the escape.

---

## 6. Phase 3 — Gated commit with abort-on-misprediction

All actions route through `commit_actions(queue)`. Each executed step is compared
against the model's prediction; the first mismatch discards the remaining plan
and returns control to theorizing with the contradicting transition added.

**[revised] Two adjustments:**

1. **Scale batch length to evidence.** Short plans while the transition log is
   thin, long ones once certified over many transitions. A fixed long batch
   wastes actions when the model is young.
2. **Drop v1's separate "predict-before-act" cheap win.** Once `step()` exists,
   the prediction *is* `step()`, and this channel already checks it. Requiring a
   written prose prediction on every action is a token tax that duplicates the
   mechanism. Keep it only as a pre-certification bootstrap.

---

## 7. Phase 4 — Representation revision

When backtest failures persist across multiple rewrite attempts, prompt the agent
to revise `encode` — what the objects are, whether hidden state exists — rather
than continuing to patch `step`.

Schema's agent spent three levels treating a block as directly steerable before
recording that the block moves when the player's **carry state** changes, a
variable appearing nowhere in the grid. Mispredictions collapsed from dozens per
level to four.

**[revised]** With `encode` as a first-class model-owned function, this is a
natural escalation rather than a new mechanism: rule revision edits `step`,
representation revision edits `encode`. The prompt escalates after K failed
`step`-only rewrites.

---

## 8. Cheap wins

| Fix | Status | Why |
| --- | --- | --- |
| No-effect memory | **in Phase 1b** | harness-owned; model cannot do it |
| Board-change report | **in Phase 1b** | harness-owned; feeds `encode` revision |
| Click-target pruning | deferred | valuable, but the model *can* derive it from `segmentation` — lower priority than harness-only capabilities |
| Oscillation guard | deferred | partly subsumed by no-effect memory + planning |
| Macro-action commit | **Phase 3** | folded into `commit_actions` batch sizing |
| Predict-before-act | **dropped post-certification** | duplicates Phase 3 |

---

## 9. Phase 5 — Trace distillation

Unchanged from v1: run a frontier model through the harness, filter to
high-RHAE episodes, SFT Qwen on the traces exported by `make traces`.

**[revised] It may need to move earlier.** If the Phase 1 measurement shows the
27B cannot induce certifiable models on more than a game or two, Phases 2–4 have
no foundation and distillation becomes the critical path rather than the finisher.
Target the representation-questioning behaviour from Phase 4 specifically.

---

## 10. Measurement discipline

Baseline **1.6002 ± 0.4475** on the public set — the standard deviation is 28% of
the mean, so most improvements measured at fewer than 20 passes are noise.

- Keep `make significance`, paired by game, `P(delta > 0) >= 0.90`
- Treat anything below ~2x the paired SE as unproven
- Hold out games; do not tune on all 25
- Iterate locally with `SIMULATE_COMPETITION_ARCADE=true`; Kaggle's ~30
  GPU-hours/week affords roughly 15–20 full validation runs

**[revised] Add a leading indicator.** RHAE is too noisy to steer Phase 1 by. Log
per-game certification statistics — did a model get induced, what fraction of
transitions did it reproduce, how often did the fallback fire — and use those to
decide whether to continue. They move long before the score does.

**[revised] Watch for changed run shape.** Certified planning replaces a steady
one-action-per-turn rhythm with long silences punctuated by macro-commits. The
`kaggle-duck` defaults (132 min/game, 28 concurrent) were tuned for the current
rhythm and will need re-checking after Phase 3.

---

## 11. Sequencing

| Stage | Work | Exit criterion |
| --- | --- | --- |
| 1 | Phase 1 + 1b (see `implementation-plan-v2-phase1.md`) | certification rate measured on >= 20 passes |
| 2 | **Go/no-go.** Certifies on >= 3 of 25 games -> continue. Below that -> Phase 5 first. | explicit decision, written down |
| 3 | Phase 2 + Phase 3 | solved-level action counts drop vs baseline |
| 4 | Phase 4 | mispredictions per level drop |
| 5 | Phase 5 | — |
| 6 | Robustness only: wall-clock guards, vLLM death handling, full arcade simulation | — |

**[revised]** v1 sequenced by calendar weeks. This sequences by measured exit
criteria, because stage 2 can send you down a completely different path and a
date cannot tell you that.

A crash on the private set is a total loss. In the final stage that risk
outweighs any marginal feature.

---

## 12. Deployment notes

The Kaggle notebook is infrastructure; the agent is frozen in
`benchmark_initial.pkl` inside an attached dataset.

- Pickles store instance state, not class code. Unpickling re-imports
  `HarnessSolver` from whatever is on `sys.path`, so rewritten method bodies and
  new tools are picked up automatically.
- **But**: a new attribute set in `__init__` will `AttributeError` against an old
  pickled `__dict__`. Every task in the implementation plan that adds agent state
  must guard reads with `getattr(self, name, default)` **and** the bundle must be
  rebuilt rather than source-swapped.
- **[revised]** v1 says to hand-edit `DATASET_SOURCES` in the notebook. Don't —
  `make kaggle-duck` sets the dataset ref from your Kaggle username. See
  `UPDATE-AND-EVALUATE.md`.
- Milestone eligibility requires open-sourcing under a permissive licence with
  the Duck attributable as third-party work.
