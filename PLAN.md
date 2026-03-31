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
| Metric | `val_bpb` (bits per byte) | `eval_loss` (cross-entropy on held-out split) |
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
- [ ] Copy datasets from `/media/johnson_ssd/finetune-setup/datasets/` to repo
- [ ] Copy `Dockerfile` from `/media/johnson_ssd/finetune-setup/Dockerfile`
- [ ] Update `.gitignore` for finetuning workflow
- [ ] Remove files not needed: `pyproject.toml`, `uv.lock`, `.python-version`, `analysis.ipynb`, `progress.png`

### Phase 2: Core Scripts
- [ ] Write `prepare.py` — dataset download + unification (from existing `prepare_datasets.py`)
- [ ] Write `train.py` — the main script with sections:
  - [ ] Section 1: CONFIG (all hyperparams as top-level constants)
  - [ ] Section 2: IMPORTS + SETUP (timing, run name)
  - [ ] Section 3: LOAD MODEL (FastLanguageModel + LoRA)
  - [ ] Section 4: DATASET (load, format to text, filter, split)
  - [ ] Section 5: TRAIN (SFTTrainer)
  - [ ] Section 6: EVALUATE (eval_loss, perplexity)
  - [ ] Section 7: INFERENCE TEST (3 prompts, decoded output)
  - [ ] Section 8: RESULTS SUMMARY + LOG (print block + append results.tsv)
  - [ ] Section 9: SAVE (LoRA adapter)

### Phase 3: Documentation
- [ ] Write `program.md` — agent instructions (setup, loop, research directions, rules)
- [ ] Write `README.md` — overview, quick start, structure
- [ ] Move old notebooks to `archive/`

### Phase 4: Test
- [ ] Run `python prepare.py` inside Docker — verify datasets created
- [ ] Run `python train.py > run.log 2>&1` inside Docker — verify end-to-end
- [ ] Verify `results.tsv` has one row appended
- [ ] Verify `outputs/lora_adapter/` exists
- [ ] Verify parseable output: `grep "^eval_loss:" run.log`
- [ ] Verify agent can extract metrics and make keep/discard decisions

### Phase 5: Commit & Push
- [ ] Initial commit on `feature/unsloth-finetune` branch
- [ ] Push to `fork` remote (`superoo7/autoresearch-cu130`)
