# Updating the Duck and evaluating it on Kaggle

The loop this document describes:

```
upstream ─▶ your fork ─▶ local change ─▶ local smoke run ─▶ Kaggle dataset version ─▶ notebook run ─▶ score ─▶ significance
```

Kaggle submission notebooks have **no internet**. The harness source therefore travels to
Kaggle as a *dataset*, not a `git clone`. Every code change you want to evaluate on Kaggle
needs a new version of that dataset, which `make kaggle-duck` builds and pushes for you.

All `make` commands run from `ARC3-Inference/` unless stated otherwise.

---

## 0. One-time setup

### 0.1 Remotes

Already configured in this checkout:

| Remote | URL | Use |
| --- | --- | --- |
| `origin` | `https://github.com/Tanapat-Riew/duck-harness.git` | your work |
| `upstream` | `https://github.com/Tufalabs/duck-harness.git` | pull Tufa's fixes |

Verify with:

```bash
git remote -v
```

### 0.2 Toolchain

Python 3.12, `uv`, and GNU `make`. On Windows run every command below from **Git Bash**
or WSL — the Makefile is POSIX shell and will not work in PowerShell or `cmd`.

Light install (viewer, scoring, significance, trace export — no GPU):

```bash
cd ARC3-Inference && uv sync --locked
```

Full install (adds the `server` extra: vLLM + Torch, multi-GB, GPU only):

```bash
cd ARC3-Inference && make install
```

### 0.3 Kaggle CLI and credentials

The CLI is **not currently installed on this machine**. Install it:

```bash
uv tool install kaggle
```

Then create an API token at <https://www.kaggle.com/settings> → *Create New Token*, and save
the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`. `KAGGLE_USERNAME` / `KAGGLE_KEY`
environment variables and a `.env` file also work — `deploy_kaggle.py` walks parent
directories looking for `.env`.

Confirm auth:

```bash
kaggle kernels list --mine --page-size 1
```

### 0.4 Output directory on Windows

`configs/inference.json` sets `experiments.root_dir` to `/shared/arc_3_results/{username}`,
a cluster path that does not exist locally. Pass `EXPERIMENTS_DIR=runs` on every local run
command, or edit that key in your own config. Every example below already does this.

---

## 1. Sync with upstream

Do this before starting any change, so you rebase onto Tufa's latest rather than resolving
conflicts after the fact.

```bash
git fetch upstream
```

```bash
git merge upstream/main
```

```bash
git push origin main
```

## 2. Branch

```bash
git switch -c improve-world-model
```

## 3. Make the change

The files that actually change agent behaviour:

| File | What lives there |
| --- | --- |
| `ARC3-Inference/inference/agent/tool_agent.py` | the turn loop, prompt assembly, world-model memory, context trimming |
| `ARC3-Inference/inference/agent/prompts.py` | every system-prompt addendum |
| `ARC3-Inference/inference/agent/python_tool_sandbox.py` | the sandbox and the globals exposed to model code |
| `ARC3-Inference/inference/utils/segmentation.py` | board segmentation the model reasons over |
| `ARC3-Inference/configs/inference.json` | sampling, context window, tool budgets, game selection |

Config-only experiments (temperature, `analyzer.tool_steps`, `analyzer.yield_seconds`,
`multimodal.context`) need no code edit — change the JSON and rerun.

## 4. Static checks

```bash
cd ARC3-Inference && make prepare-ci
```

Runs Ruff plus the test suite. Fix failures here before spending GPU minutes.

## 5. Local smoke run

Never push an untested change to Kaggle — a bad run costs a GPU session. Play one game
locally first.

**Through OpenRouter** (no local GPU needed — the practical option on a laptop):

```bash
export OPENROUTER_API_KEY=<your-key>
```

```bash
cd ARC3-Inference && CONFIG_PATH=configs/inference.openrouter.json make interactive GAME=ar25 N_PASSES=1 MAX_RUNTIME_MINUTES=10 EXPERIMENTS_DIR=runs
```

**Through a local vLLM server** (GPU machine):

```bash
cd ARC3-Inference && make server
```

```bash
cd ARC3-Inference && make interactive GAME=ar25 N_PASSES=1 MAX_RUNTIME_MINUTES=10 EXPERIMENTS_DIR=runs
```

**Against TAAF's competition Arcade simulator** — catches submission-Arcade problems without
burning a Kaggle rerun. Inline only:

```bash
cd ARC3-Inference && make interactive GAME=[] GAME_TAGS=official SIMULATE_COMPETITION_ARCADE=true COMPETITION_CLONE_RUNS=110 N_PASSES=1 EXPERIMENTS_DIR=runs
```

## 6. Read the run before trusting the number

```bash
cd ARC3-Inference && make view VIEW_RUN_DIR=runs/<your-run-dir>
```

Opens at `http://127.0.0.1:8011` with per-game boards, actions, rewards, level transitions,
and the full duck transcript. For a prompt or memory change, read the transcript — a score
that moved for the wrong reason looks identical to one that moved for the right reason.

Useful raw artifacts in the same run directory:

- `prompts/<game>.log` — the most recent full model call, verbatim
- `artifacts/<game>_events.jsonl` — append-only viewer events
- `<game>_requests.jsonl` — every request/response, when `analyzer.save_request_logs` is on

## 7. Commit and push

```bash
git add -A && git commit -m "Describe the behaviour change"
```

```bash
git push -u origin improve-world-model
```

GitHub is your history and backup. It is **not** how code reaches Kaggle — that is step 8.

## 8. Build the Kaggle bundle, dry run first

```bash
cd ARC3-Inference && make kaggle-duck KAGGLE_DRY_RUN=true RUN_NAME=duck-test EXPERIMENTS_DIR=runs
```

This writes the bundle and metadata but makes **no** Kaggle API calls. Inspect what was
staged under the printed run directory:

```
<run-dir>/kaggle/source-dataset/
    taaf-kaggle-bundle.json     marker the notebook searches for
    dataset-metadata.json       dataset ref + title
    deploy_target.pkl           pickled deployment target
    benchmark_initial.pkl       pickled benchmark
    src/<repo>/                 snapshot of ARC3-Inference and TAAF
<run-dir>/kaggle/kernel/
    kernel-metadata.json        notebook slug, attached datasets, accelerator
    taaf_kaggle_run*.ipynb      rendered notebook
```

Confirm your edited files are inside `src/`. If a repo is missing, add it to
`deployment.source_repos` in `configs/inference.json`.

## 9. Push to Kaggle

```bash
cd ARC3-Inference && make kaggle-duck RUN_NAME=duck-2026-08-13 EXPERIMENTS_DIR=runs
```

That single variable is usually enough:

- `KAGGLE_KERNEL_SLUG` defaults to `taaf-<RUN_NAME>`
- `KAGGLE_DATASET_REF` defaults to `<your-kaggle-username>/taaf-kaggle-source`
- the notebook is created **private** unless you pass `KAGGLE_PUBLIC=true`

Re-running with the same refs uploads a **new dataset version** and pushes a new notebook
version — that is the normal update path, not an error.

Override names explicitly when you want one dataset per experiment:

```bash
cd ARC3-Inference && make kaggle-duck RUN_NAME=duck-2026-08-13 KAGGLE_KERNEL_SLUG=taaf-duck-2026-08-13 KAGGLE_DATASET_REF=<your-kaggle-username>/taaf-src-duck-2026-08-13 EXPERIMENTS_DIR=runs
```

Actual defaults baked into the `kaggle-duck` target (`Makefile:391`):

| Variable | Value |
| --- | --- |
| `MODEL` | `local` (vLLM started inside the notebook) |
| `N_PASSES` | 1 |
| `CONCURRENT_JOBS` | 28 |
| `MAX_RUNTIME_MINUTES` | 132 per game |
| `MAX_EXPERIMENT_RUNTIME_MINUTES` | 540 |
| `ANALYZER_TIMEOUT` | 900 |
| `KAGGLE_ACCELERATOR` | `NvidiaRtxPro6000` |

## 10. Watch the run and pull results

Block until the notebook finishes and pull its output into the run directory:

```bash
cd ARC3-Inference && make kaggle-duck RUN_NAME=duck-2026-08-13 DEPLOYMENT_WAIT=true EXPERIMENTS_DIR=runs
```

Or poll manually:

```bash
kaggle kernels status <your-kaggle-username>/taaf-duck-2026-08-13
```

```bash
kaggle kernels output <your-kaggle-username>/taaf-duck-2026-08-13 -p ./kaggle-output
```

## 11. Score the run

```bash
cd ARC3-Inference && make score_run SCORE_RUN_DIR=<run-dir> SCORE_OUTPUT_PATH=docs/candidate-score.json
```

Writes `evaluation.json` plus the lightweight `score.json` format the significance check
consumes. To score several runs selected by config, edit `configs/eval.json` and run
`make eval`.

## 12. Decide whether it actually helped

```bash
cd ARC3-Inference && make significance BASELINE_SCORE=docs/current-best-score.json CANDIDATE_SCORE=docs/candidate-score.json
```

Games are the paired unit, repeated trials are averaged within a game, and the bar is:

```
P(true_delta > 0 | results) >= 0.90
```

The output also reports win rate, a bootstrap 90% interval, and TAAF paired p-values. The
check validates runtime budget, hardware, dataset metadata, and trial counts first — compare
runs of the same shape, or the answer is meaningless.

**Get a baseline before your first change.** Run steps 8–11 on unmodified `main`, save the
result as `docs/current-best-score.json`, and only replace it when a candidate clears the
threshold. Without that file, step 12 has nothing to compare against.

---

## Gotchas

**Kaggle datasets owned by other accounts.** `inference/framework/kaggle.py:10-11` hardcodes
`driessmit1/arc3-vllm-h100-wheelhouse-v3` (vLLM wheels) and
`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` (model weights). They are attached to your
notebook automatically and work while they remain public. If they disappear, or you switch
models, copy them to your own account and edit `DuckKaggleVllmConfig`.

**The root notebook is generated output.** `taaf-duck-harness-kaggle-share.ipynb` is a
rendered artifact of a past deploy, with `jeroencottaar/taaf-kaggle-source-share` baked into
its `DATASET_SOURCES`. Hand-editing it will not change what `make kaggle-duck` produces.
Edit the template at `tufa-arc-agi-framework/src/taaf/kaggle/taaf_kaggle_run_share.ipynb`
instead, or just regenerate.

**`ARC3-Inference/README.md` is stale on the Kaggle defaults.** It describes 16 games, 16
concurrent jobs, 75 minutes per game, and a 90-minute cap. The Makefile actually sets 25
games, 28 concurrent jobs, 132 minutes, and 540. Trust the Makefile.

**GPU assertion.** The notebook's setup script asserts the GPU matches `KAGGLE_GPU_TYPE`
(default `rtx-pro-6000`) and `KAGGLE_GPU_COUNT`. A mismatched Kaggle accelerator fails fast
rather than running slowly.

**Prompt-affecting env vars are frozen at bundle time.** `duck_kaggle_setup_command()`
embeds the launcher's resolved `LOCAL_ANALYZER_*` and `MULTIMODAL_*` values into the
notebook's setup script. Changing a config value means rebuilding the bundle — a notebook
rerun alone will not pick it up.

---

## Command reference

| Goal | Command |
| --- | --- |
| Sync upstream | `git fetch upstream && git merge upstream/main` |
| Lint + tests | `make prepare-ci` |
| One game locally | `make interactive GAME=ar25 N_PASSES=1 MAX_RUNTIME_MINUTES=10 EXPERIMENTS_DIR=runs` |
| Inspect a run | `make view VIEW_RUN_DIR=runs/<run>` |
| Stage Kaggle bundle only | `make kaggle-duck KAGGLE_DRY_RUN=true RUN_NAME=<name> EXPERIMENTS_DIR=runs` |
| Push to Kaggle | `make kaggle-duck RUN_NAME=<name> EXPERIMENTS_DIR=runs` |
| Push and wait | add `DEPLOYMENT_WAIT=true` |
| Score a run | `make score_run SCORE_RUN_DIR=<run> SCORE_OUTPUT_PATH=docs/candidate-score.json` |
| Compare to baseline | `make significance BASELINE_SCORE=... CANDIDATE_SCORE=...` |
| Export traces | `make traces` |
| Probe the model directly | `make chat PROMPT="..."` |
