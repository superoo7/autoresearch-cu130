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

### Phase 1: Setup
- [x] Copy datasets from `/media/johnson_ssd/finetune-setup/datasets/` to repo
- [x] Copy `Dockerfile` from `/media/johnson_ssd/finetune-setup/Dockerfile`
- [x] Update `.gitignore` for finetuning workflow
- [x] Remove files not needed: `pyproject.toml`, `uv.lock`, `.python-version`, `analysis.ipynb`, `progress.png`

### Phase 2: Core Scripts
- [x] Write `prepare.py` — dataset download + unification (from existing `prepare_datasets.py`)
- [x] Write `train.py` — the main script with sections:
  - [x] Section 1: CONFIG (all hyperparams as top-level constants)
  - [x] Section 2: IMPORTS + SETUP (timing, run name)
  - [x] Section 3: LOAD MODEL (FastLanguageModel + LoRA)
  - [x] Section 4: DATASET (load, format to text, filter, split)
  - [x] Section 5: TRAIN (SFTTrainer)
  - [x] Section 6: EVALUATE (eval_loss, perplexity)
  - [x] Section 7: INFERENCE TEST (3 prompts, decoded output)
  - [x] Section 8: RESULTS SUMMARY + LOG (print block + append results.tsv)
  - [x] Section 9: SAVE (LoRA adapter)
- [ ] Add output quality checklist scoring to train.py (Section 6b):
  - [ ] Define CHECKLIST questions in CONFIG
  - [ ] After inference, score each output against checklist (yes/no per item)
  - [ ] Compute checklist_score = % passed across all prompts
  - [ ] Print `checklist_score:` in summary block
  - [ ] Log checklist_score to results.tsv

### Phase 3: Documentation
- [x] Write `program.md` — agent instructions (setup, loop, research directions, rules)
- [x] Write `README.md` — overview, quick start, structure
- [x] Move old notebooks to `archive/`
- [ ] Update `program.md` with checklist scoring rules for the agent loop

### Phase 4: Test
- [ ] Run `python prepare.py` inside Docker — verify datasets created
- [x] Run `python train.py > run.log 2>&1` inside Docker — verify end-to-end (running now)
- [ ] Verify `results.tsv` has one row appended
- [ ] Verify `outputs/lora_adapter/` exists
- [ ] Verify parseable output: `grep "^eval_loss:\|^checklist_score:" run.log`
- [ ] Verify agent can extract metrics and make keep/discard decisions

### Phase 5: Checklist Scoring (from autoresearch skill pattern)
- [ ] Add CHECKLIST config to train.py CONFIG section:
  - [ ] Define 5 binary yes/no eval questions for reasoning output quality
  - [ ] Add `CHECKLIST_PASS_THRESHOLD = 0.6` (quality gate)
- [ ] Add `score_output()` function to train.py:
  - [ ] Takes generated text, returns dict of {eval_name: True/False}
  - [ ] Uses simple string/pattern checks (no LLM judge needed for v1)
  - [ ] e.g. check for `<think>` tags, step-by-step markers, answer outside tags
- [ ] Add Section 6b to train.py: score all test prompt outputs against checklist
- [ ] Print `checklist_score:` in summary block
- [ ] Add `checklist_score` column to results.tsv
- [ ] Create `changelog.md` template — train.py appends after each run

### Phase 6: Agent Loop Refinements
- [ ] Update `program.md` with:
  - [ ] Checklist scoring rules (keep only if eval_loss improved AND checklist >= 60%)
  - [ ] Failure analysis step (read failing outputs before choosing next experiment)
  - [ ] Changelog writing rules (append to changelog.md after each experiment)
  - [ ] Ceiling detection (stop at 95%+ checklist for 3 consecutive runs)
  - [ ] One-change-at-a-time rule (explicit)
- [ ] Copy `eval-guide.md` to `references/` for agent reference
- [ ] Add baseline-first rule to program.md (run unchanged config first)

### Phase 7: Polish & Ship
- [ ] Copy autoresearch skill zip to `archive/autoresearch-skill/` for reference
- [ ] Commit all changes on `feature/unsloth-finetune` branch
- [ ] Push to `fork` remote (`superoo7/autoresearch-cu130`)
- [ ] Wait for Phase 4 test run to complete, verify results
- [ ] Fix any issues found during test run
