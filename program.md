# autoresearch-cu130

Autonomous SFT finetuning research for Qwen3.5 with LoRA on RunPod (RTX PRO 6000 Blackwell, 96GB VRAM).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr07`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — dataset preparation. Do not modify.
   - `train.py` — the file you modify. LoRA config, hyperparameters, dataset filtering.
   - `changelog.md` — history of all experiments and results.
4. **Verify data exists**: Check that `datasets/combined_reasoning/` exists. If not, run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment finetunes Qwen3.5-9B with LoRA on ~5,100 reasoning examples (after length filtering). You launch it as: `.venv/bin/python train.py > run.log 2>&1`

**Important:** Do NOT use `uv run train.py` — this branch has no `pyproject.toml`. Always use `.venv/bin/python`. If the venv is missing or broken, run `bash setup.sh`.

Training uses **completion-only masking** — loss is only computed on the assistant response, not on user/system tokens. This focuses learning on reasoning + answer generation.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. The CONFIG section at the top has all hyperparameters: LoRA rank/alpha/targets, learning rate, epochs, batch size, sequence length, dataset filtering, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the dataset preparation logic.
- Install new packages or add dependencies.
- Modify the evaluation logic (eval_loss computed by HuggingFace Trainer).

**The goal is simple: get the lowest eval_loss.** Lower eval_loss = better reasoning ability. Perplexity = exp(eval_loss) is also reported for intuition.

**VRAM** is a soft constraint. The RTX PRO 6000 has 96GB discrete VRAM. Most configs will fit comfortably. Monitor `peak_vram_mb` in results.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
eval_loss:        1.234567
perplexity:       3.4356
train_loss:       1.123456
checklist_score:  0.67
checklist_detail: 6/9
peak_vram_mb:     45060.2
train_time_min:   15.3
dataset_rows:     3050
lora_rank:        16
lora_alpha:       16
learning_rate:    0.0001
num_epochs:       1
seq_length:       2048
batch_size:       1
grad_accum:       8
completion_only:  True
```

You can extract the key metrics from the log file:

```
grep "^eval_loss:\|^checklist_score:" run.log
```

## Research directions (priority order)

### Prior results (from DGX Spark, 25 experiments)
- Best eval_loss: **0.908** (exp24, Qwen3.5-9B, 700 steps, but checklist 0/9)
- Best checklist: **0.67** (exp14, Qwen3.5-4B, 700 steps, eval_loss 0.985)
- Best PinchBench: **78.0%** (exp12, Qwen3.5-4B, 600 steps)
- See `changelog.md` for full history

### Next priorities for 9B model
1. **Get checklist passing on 9B** — exp24 had 0/9 checklist (model uses prose "Thinking Process:" instead of <think> tags). More steps (exp25 tried 900) or different LR may help.
2. **Sequence length sweep** — 2048 → 4096. With 96GB VRAM, 4096 is feasible. More data preserved intact.
3. **Learning rate tuning** — try 5e-5, 1e-4, 1.5e-4, 2e-4 on 9B.
4. **LoRA rank sweep** — 16 → 32 → 64. 9B has more capacity; higher rank may help.
5. **Epoch/steps tuning** — find the sweet spot between underfitting (too few steps) and NaN (750+ was NaN boundary on 4B).
6. **Dataset filtering** — filter by source, length, or quality.

### Known constraints from prior experiments
- LoRA alpha must equal rank for Qwen3.5 (exp15: alpha=2x rank → catastrophic)
- Attention-only LoRA targets regress badly (exp11: eval_loss +0.05)
- LR below 5e-5 underfits at 600 steps (exp8)
- Grad accum 16 is too few optimizer updates (exp18: catastrophic)
- 750 steps was NaN boundary on 4B (exp5, exp22) — may differ on 9B

## Logging results

When an experiment is done, the script auto-appends to `results.tsv`. You can also manually log notes.

Read the latest results: `tail -5 results.tsv`

## The experiment loop
The experiment runs on a dedicated branch (e.g. autoresearch/apr07).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. **Analyze failures first.** Read the CHECKLIST RESULTS and INFERENCE TEST outputs from the last run.log. Which checklist items fail? What do the failing outputs look like? Form a hypothesis before changing anything.
3. **Change ONE thing** in the CONFIG section of `train.py`. Never change multiple hyperparams at once — you won't know what helped.
4. git commit with a message describing what you changed and why
5. Run: `.venv/bin/python train.py > run.log 2>&1`
6. Read results: `grep "^eval_loss:\|^checklist_score:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback.
8. **Keep or revert:**
   - eval_loss improved AND checklist_score >= 0.60 → **KEEP**
   - eval_loss improved BUT checklist_score < 0.60 → **REVERT** (quality gate failed)
   - eval_loss same or worse → **REVERT** via `git reset --hard HEAD~1`
9. **Log to changelog.md**: Append what you changed, why, and what happened (see format below)
10. **Upload to HuggingFace** (if HF_UPLOAD=True in train.py): LoRA adapter + GGUF are auto-uploaded after training. The user can download on their DGX Spark for testing with ollama/PinchBench.
11. Repeat from step 1

**Changelog format** (append after each experiment):
```markdown
## Experiment [N] — [keep/discard]
**eval_loss:** [X.XXXX] | **checklist:** [X.XX] | **Change:** [one sentence]
**Reasoning:** [why you expected this to help]
**Result:** [what actually happened]
```

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
