# autoresearch-cu130

Autonomous SFT finetuning research for GPT-OSS 20B with LoRA on DGX Spark.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar31`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — dataset preparation. Do not modify.
   - `train.py` — the file you modify. LoRA config, hyperparameters, dataset filtering.
4. **Verify data exists**: Check that `datasets/combined_reasoning/` exists. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment finetunes GPT-OSS 20B with LoRA on 3,209 curated reasoning examples. You launch it as: `python train.py > run.log 2>&1`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. The CONFIG section at the top has all hyperparameters: LoRA rank/alpha/targets, learning rate, epochs, batch size, sequence length, dataset filtering, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the dataset preparation logic.
- Install new packages or add dependencies.
- Modify the evaluation logic (eval_loss computed by HuggingFace Trainer).

**The goal is simple: get the lowest eval_loss.** Lower eval_loss = better reasoning ability. Perplexity = exp(eval_loss) is also reported for intuition.

**VRAM** is a soft constraint. The DGX Spark has 128GB unified memory. Some increase is acceptable for meaningful eval_loss gains.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
eval_loss:        1.234567
perplexity:       3.4356
train_loss:       1.123456
peak_vram_mb:     45060.2
train_time_min:   15.3
dataset_rows:     3050
lora_rank:        16
lora_alpha:       32
learning_rate:    0.0002
num_epochs:       3
seq_length:       4096
batch_size:       1
grad_accum:       4
```

You can extract the key metric from the log file:

```
grep "^eval_loss:" run.log
```

## Research directions (priority order)

### 1. LoRA Rank Sweep
Try ranks: 4, 8, 16, 32, 64. Higher rank = more capacity but slower.
Current: 16. Hypothesis: 32 may be the sweet spot for this dataset size.

### 2. Learning Rate Tuning
Try: 5e-5, 1e-4, 2e-4, 3e-4, 5e-4.
Current: 2e-4. Watch for instability at higher rates.

### 3. Sequence Length Impact
Try: 2048, 4096, 8192. Longer keeps more data intact but uses more VRAM.
Check how many rows are dropped at each length.

### 4. Epoch Count vs Overfitting
Try: 1, 2, 3, 5 epochs. Monitor train_loss vs eval_loss gap.
With 3,209 rows, overfitting is a real risk beyond 3 epochs.

### 5. Dataset Filtering
- Filter by source (train only on opus-4.6, or only qwen, etc.)
- Filter by token length (drop very short examples)
- Filter by category/difficulty if metadata available

### 6. LoRA Target Modules
- Attention-only: q_proj, k_proj, v_proj, o_proj
- Full (default): also gate_proj, up_proj, down_proj
- Try adding/removing targets

### 7. Advanced
- Gradient accumulation: 4, 8, 16 (larger effective batch)
- Weight decay: 0.0, 0.01, 0.1
- Warmup ratio: 0.0, 0.05, 0.1
- LoRA alpha ratio: 1x, 2x, 4x rank
- Optimizer: adamw_8bit vs adamw_torch

## Logging results

When an experiment is done, the script auto-appends to `results.tsv`. You can also manually log notes.

Read the latest results: `tail -5 results.tsv`

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify the CONFIG section of `train.py` with an experimental idea
3. git commit
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^eval_loss:\|^perplexity:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback.
7. If eval_loss improved (lower), keep the commit
8. If eval_loss is equal or worse, `git reset --hard HEAD~1` to revert

**Crashes**: If a run crashes, check if it's a simple fix (typo, OOM). For OOM, reduce MAX_SEQ_LENGTH or BATCH_SIZE. If fundamentally broken, revert and try something else.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
