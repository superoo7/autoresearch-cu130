# PLAN: autoresearch → unsloth finetuning

Convert autoresearch from a pre-training research loop into an SFT finetuning research loop using Unsloth + LoRA on DGX Spark.

## Context

**Original autoresearch**: An agent modifies `train.py` (raw PyTorch GPT pre-training), trains for 5 minutes, checks `val_bpb`, keeps/discards, repeats. The agent experiments with model architecture, optimizer, hyperparams.

**This fork (autoresearch-cu130)**: Same autonomous loop, but for **finetuning** a pre-trained model (GPT-OSS 20B) on curated reasoning data using Unsloth's SFT + LoRA. The agent experiments with LoRA config, training hyperparams, dataset composition, and sequence length.

| | Original (pre-training) | This fork (finetuning) |
|---|---|---|
| Task | Train GPT from scratch | SFT finetune GPT-OSS 20B |
| What agent modifies | Model architecture, optimizer, training loop | LoRA config, hyperparams, dataset filtering |
| Framework | Raw PyTorch | Unsloth + TRL (SFTTrainer) |
| Runtime | `uv run train.py` (uv venv) | `python train.py` (Docker: `unsloth-dgx-spark`) |
| Metric | `val_bpb` (bits per byte) | `eval_loss` + checklist score on outputs |
| Time budget | Fixed 5 min wall clock | Fixed by epoch count (small dataset) |
| Base model | None (from scratch) | `unsloth/gpt-oss-20b` (20B MoE, 4-bit) |
| GPU | H100 | DGX Spark GB10 (128GB unified memory) |
| CUDA | 12.8 | 13.0 |
| Data | HF parquet shards (web text) | 3,209 curated reasoning examples |

## Agent Search Space

Instead of modifying model architecture, the agent tunes:

- **LoRA**: rank (4–64), alpha, target modules (attention-only vs full)
- **Training**: learning rate (5e-5 to 5e-4), scheduler, optimizer, epochs, warmup
- **Data**: sequence length (2048/4096/8192), filter by source/length/category, eval split size
- **Efficiency**: batch size, gradient accumulation, 4bit vs 16bit

## Dataset

3 reasoning datasets unified into `datasets/combined_reasoning/` (3,209 rows):

| Source | Rows | Format |
|---|---|---|
| Opus-4.6-Reasoning-3000x-filtered | 2,326 | `<think>` wrapped thinking + solution |
| claude-4.5-opus-high-reasoning-250x | 250 | Chat messages with `<think>` tags |
| Qwen3.5-reasoning-700x | 633 | Converted human/gpt → user/assistant |

All normalized to: `{"messages": [{role, content}, ...], "source": "...", "category": "...", "difficulty": "..."}`

## Evaluation: Two-Layer Scoring

Inspired by [the autoresearch method for skill improvement](https://aisolo.beehiiv.com/), we use **two layers of evaluation** — not just loss, but also a checklist on actual model outputs.

### Layer 1: eval_loss (quantitative, fast)
- Cross-entropy on 5% held-out split
- Computed automatically by HF Trainer
- Good for comparing hyperparameter changes
- But: low loss doesn't guarantee good reasoning outputs

### Layer 2: Output Quality Checklist (qualitative, slower)
After training, the model generates answers to test prompts. Each output is scored against a yes/no checklist:

1. **"Does the response use `<think>` tags to show reasoning before answering?"** — catches models that skip chain-of-thought
2. **"Does the thinking section show step-by-step breakdown?"** — catches shallow reasoning like restating the question
3. **"Is the final answer outside the `<think>` tags and clearly stated?"** — catches models that bury the answer in reasoning
4. **"Is the reasoning logically sound (no contradictions or skipped steps)?"** — catches hallucinated logic
5. **"Does the response actually answer the question asked?"** — catches tangential responses

The checklist score = % of checks passed across all test prompts. The agent optimizes for **lowest eval_loss** as the primary metric, but uses checklist score as a **quality gate** — a change that lowers eval_loss but drops checklist score below 60% should be reverted.

### How it works in the loop
```
1. Agent modifies train.py CONFIG
2. Trains the model
3. eval_loss computed automatically (Layer 1)
4. Model generates on test prompts, scored against checklist (Layer 2)
5. Both metrics logged to results.tsv
6. Keep if eval_loss improved AND checklist >= 60%
7. Revert otherwise
```

The checklist can be scored by the agent itself (self-eval on the generated text) or by a separate judge call. Start with self-eval for speed.

3-6 checklist items is the sweet spot — more and the model starts gaming individual checks at the expense of overall quality.

## Learnings from Autoresearch Skill Pattern

The [autoresearch skill](https://www.dropbox.com/scl/fi/57v11vtj9gzqz10ybv7or/autoresearch.zip) (saved at `archive/autoresearch-skill/`) provides a battle-tested pattern for autonomous improvement loops. Key takeaways for our finetuning setup:

### Pattern: Binary Evals, Not Scales
Every eval must be yes/no. No "rate 1-10". Binary evals give reliable signal; scales compound variability. Our checklist items must be binary.

### Pattern: One Change at a Time
Never change 5 hyperparams at once. Change one thing, measure, keep/discard. This is critical for the agent loop — if you change LoRA rank AND learning rate, you don't know what helped.

### Pattern: Analyze Failures Before Mutating
The agent should look at *which* checklist items fail most, read the actual failing outputs, then form a hypothesis about what to change. Not random exploration.

### Pattern: Changelog as the Most Valuable Artifact
Every experiment gets logged with: what changed, why, what happened. This changelog lets future agents (or smarter models) pick up where the last one left off.

### Pattern: Baseline First
Always run the unchanged config first. Never modify before measuring the starting point.

### Pattern: Ceiling Detection
Stop when you hit 95%+ on checklist for 3 consecutive runs. Diminishing returns.

### Pattern: Dashboard (optional, future)
The skill creates a live HTML dashboard with auto-refresh. For our finetuning use case, this could show:
- eval_loss trend over experiments
- checklist pass rates per item
- keep/discard status per experiment

### What We Can Reuse Directly
1. **Binary eval format** — adapt for reasoning output quality (see Layer 2 above)
2. **results.tsv format** — already implemented, add checklist columns
3. **changelog.md** — add to train.py output: after each run, append what was changed and why
4. **Failure analysis pattern** — agent reads failing outputs before deciding next experiment
5. **eval-guide.md** — copy to `references/` as guidance for writing good checklist items

## Verified Compatibility

Tested inside `unsloth-dgx-spark` Docker container:
- torch: 2.10.0a0+b558c986e8.nv25.11 ✓
- CUDA: 13.0, GPU: NVIDIA GB10 ✓
- unsloth, trl, datasets, transformers 4.56.2 ✓
- `python train.py` works (no `uv run` needed, deps in Docker)

## Target File Structure

```
autoresearch-cu130/
├── prepare.py          # PROTECTED — downloads & unifies datasets
├── train.py            # Agent modifies this — LoRA config, hyperparams, data filtering
├── program.md          # Human-written research directions for the agent
├── Dockerfile          # Build unsloth-dgx-spark image
├── README.md           # Overview + quick start
├── .gitignore          # Ignore datasets/, outputs/, cache
├── PLAN.md             # This file
├── results.tsv         # Auto-appended by train.py (not gitignored)
├── datasets/           # Gitignored — created by prepare.py
│   └── combined_reasoning/
└── archive/            # Old notebooks for reference
```

## Files Changed from Original Autoresearch

| File | Action | Notes |
|---|---|---|
| `train.py` | **Rewrite** | Replace raw PyTorch pre-training with Unsloth SFT+LoRA finetuning |
| `prepare.py` | **Rewrite** | Replace data download/tokenizer with dataset unification script |
| `program.md` | **Rewrite** | Finetuning-specific research directions |
| `Dockerfile` | **Add** | Docker image with CUDA 13.0, unsloth, trl |
| `README.md` | **Rewrite** | New quick start for Docker + finetuning |
| `.gitignore` | **Update** | Add datasets/, outputs/, unsloth cache |
| `pyproject.toml` | **Remove** | Not needed (deps in Docker) |
| `uv.lock` | **Remove** | Not needed |
| `.python-version` | **Remove** | Not needed |
| `analysis.ipynb` | **Remove** | Move to archive/ if needed later |
| `progress.png` | **Remove** | Not relevant |

---

## TODO

### Phase 1: Setup — DONE
- [x] Copy datasets to repo (symlinked)
- [x] Copy `Dockerfile`
- [x] Update `.gitignore`
- [x] Remove unneeded files (`pyproject.toml`, `uv.lock`, `.python-version`, etc.)

### Phase 2: Core Scripts — DONE
- [x] Write `prepare.py` — dataset download + unification
- [x] Write `train.py` — CONFIG, model loading, dataset prep, training, eval, inference, results logging, save
- [x] Add checklist scoring to train.py (3 binary evals: think tags, step-by-step, answer outside tags)
- [x] Add `checklist_score` to summary block and results.tsv
- [x] Add quality gate threshold (CHECKLIST_PASS_THRESHOLD = 0.6)

### Phase 3: Documentation — DONE
- [x] Write `program.md` — agent loop with checklist rules, failure analysis, one-change-at-a-time, changelog format, baseline-first, ceiling detection
- [x] Write `README.md`
- [x] Move old notebooks to `archive/`
- [x] Copy `eval-guide.md` to `references/`
- [x] Archive autoresearch skill zip to `archive/autoresearch-skill/`

### Phase 4: Test — IN PROGRESS
- [x] Run `python train.py > run.log 2>&1` inside Docker (running now, ~16% epoch 1)
- [ ] Verify `results.tsv` has one row appended after run completes
- [ ] Verify `outputs/lora_adapter/` exists
- [ ] Verify parseable output: `grep "^eval_loss:\|^checklist_score:" run.log`
- [ ] Verify checklist PASS/FAIL output is sensible
- [ ] Fix any issues found during test run

### Phase 5: Remaining
- [ ] Create `changelog.md` template (agent appends after each experiment)
- [ ] Final commit + push after test run passes
