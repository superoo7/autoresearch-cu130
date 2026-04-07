# Experiment Changelog — apr01

## PinchBench Results (automated-only suite, 9 tasks)

| Model | Exp | Overall | Tokens | Requests | BASIC | CALENDAR | CODING | COMPREHENSION | CONTEXT | FILE_OPS | RESEARCH |
|-------|-----|---------|--------|----------|-------|----------|--------|---------------|---------|----------|----------|
| qwen3.5:4b (base) | — | 72.8% | 371,878 | 28 | 100% | 83.3% | 85.7% | 0% | 100% | 61.9% | 100% |
| gemma4:e4b | — | 39.7% | 432,091 | 20 | 100% | 0% | 0% | 0% | 100% | 19.0% | 100% |
| qwen3.5-4b-ft exp4 | 4 | 75.9% | 3,231,561 | 71 | 100% | 83.3% | 100% | 0% | 100% | 66.7% | 100% |
| qwen3.5-4b-ft exp12 | 12 | 78.0% | 1,303,683 | 46 | 100% | 83.3% | 0% | 33.3% | 100% | 95.2% | 100% |
| qwen3.5-4b-ft exp14 | 14 | 66.7% | 3,314,983 | 72 | 100% | 0% | 100% | 0% | 100% | 66.7% | 100% |

**Notes:**
- gemma4:e4b struggled with tool use — often responded with text instead of calling tools
- All models failed comprehension (PDF analysis) and search-and-replace tasks
- PinchBench has only 9 automated tasks — a single task flip swings score ~11%. Use as directional signal, not absolute metric
- exp14 has best eval_loss (0.985) but worst PinchBench — more training steps can make model verbose/loopy in agent tasks
- exp12 remains the best PinchBench performer (78.0%) with efficient token usage (1.3M)

---

## Experiment 0 — baseline (keep)
**eval_loss:** 1.0390 | **checklist:** 0.11 | **Change:** None (baseline)
**Reasoning:** Establish baseline with default config: rank=16, alpha=16, lr=2e-4, 300 steps, seq=2048
**Result:** eval_loss=1.039, but checklist only 1/9 — model rarely produces <think> tags. Quality gate fails (11% < 60%). This is expected for a short 300-step run; the model hasn't learned the reasoning format yet.

## Experiment 1 — keep
**eval_loss:** 1.0334 | **checklist:** 0.11 (false negative) | **Change:** MAX_STEPS 300→600
**Reasoning:** Model only saw 16% of 1 epoch at 300 steps, needed more exposure
**Result:** eval_loss improved (1.039→1.033). Checklist 1/9 but this is a BUG — the model IS producing good reasoning with </think> tags, but the opening <think> is injected by the chat template and not in the response slice. Fixed checklist to prepend <think> when missing. Also added RESUME_FROM_CHECKPOINT config.

## Experiment 2 — discard
**eval_loss:** 1.0355 | **checklist:** 0.00 | **Change:** LoRA rank 16→32, alpha 16→32
**Reasoning:** More LoRA capacity might help learn reasoning patterns
**Result:** eval_loss regressed (1.033→1.036) and checklist 0/9 — model stopped using </think> tags entirely, switched to prose "Thinking Process:" format. Higher rank hurt. Reverted.

## Experiment 3 — keep
**eval_loss:** 1.0289 | **checklist:** 0.33 | **Change:** LR 2e-4 → 1e-4
**Reasoning:** Lower LR for more stable learning of reasoning format
**Result:** eval_loss improved (1.033→1.029), checklist 3/9 (up from 1/9). Trajectory positive.

## Experiment 4 — discard
**eval_loss:** 1.0404 | **checklist:** 1.00 (9/9!) | **Change:** MAX_STEPS 600→900
**Reasoning:** More steps to push checklist past 60%
**Result:** Checklist perfect but eval_loss regressed — overfitting. Reverted.
**PinchBench:** 75.9% score | 3,231,561 tokens | 71 requests (coding 85.7%→100%, file_ops 61.9%→66.7% vs base, but ~8.7x more tokens)

## Experiment 5 — discard
**eval_loss:** NaN (1.018@step700) | **checklist:** 1.00 | **Change:** MAX_STEPS 600→750
**Reasoning:** Split difference between 600 and 900
**Result:** Final eval NaN bug. Mid-training showed best eval_loss ever (1.018). Reverted.

## Experiment 6 — discard
**eval_loss:** 1.0511 | **checklist:** 0.67 | **Change:** MAX_STEPS 600→700, SAVE_STEPS→350
**Reasoning:** Try 700 steps based on exp5's mid-training result
**Result:** Checklist passes gate (67%) for first time but eval_loss regressed badly (1.029→1.051). Reverted.

## Experiment 7 — discard
**eval_loss:** 1.0359 | **checklist:** 0.33 | **Change:** LR scheduler cosine → linear
**Reasoning:** Linear decay might provide more stable learning than cosine
**Result:** eval_loss regressed (1.029→1.036). Cosine is better. Reverted.

## Experiment 8 — discard
**eval_loss:** 1.0642 | **checklist:** 0.33 | **Change:** LR 1e-4 → 5e-5
**Reasoning:** Halving LR again — exp3 showed lower LR helped
**Result:** eval_loss regressed badly (1.029→1.064). Too low — model underfitting at 600 steps. Reverted.

## Experiment 9 — discard
**eval_loss:** 1.0445 | **checklist:** 0.67 | **Change:** LoRA rank 16→8, alpha 16→8
**Reasoning:** Lower rank might generalize better since rank 32 hurt
**Result:** Checklist passes gate (67%) but eval_loss regressed (1.029→1.045). Rank 8 lacks capacity. Reverted.

## Experiment 10 — discard
**eval_loss:** 12.6203 | **checklist:** 0.00 | **Change:** MAX_SEQ_LENGTH 2048→4096
**Reasoning:** Longer sequences preserve more reasoning data intact
**Result:** Catastrophic. Dataset tripled (5133→14961 rows) and eval_loss exploded. 600 steps is far too few for the larger dataset. Reverted.

## Experiment 11 — discard
**eval_loss:** 1.0800 | **checklist:** 0.33 | **Change:** LoRA targets attention-only (removed gate/up/down_proj)
**Reasoning:** Fewer trainable params may reduce overfitting and focus on reasoning
**Result:** eval_loss regressed significantly (1.029→1.080). MLP projections are essential for learning. Reverted.

## Experiment 12 — keep
**eval_loss:** 1.0168 | **checklist:** 0.33 | **Change:** Gradient accumulation 4→8 (effective batch 8)
**Reasoning:** Larger effective batch for more stable gradients
**Result:** Best eval_loss yet (1.029→1.017), a major improvement. Checklist 3/9 (same as exp3) — Prompt 1 passes all checks, Prompts 2-3 produce no </think> tags. Keeping because eval_loss gain is substantial and checklist is same level as previous kept state.
**PinchBench:** 78.0% score | 1,303,683 tokens | 46 requests (FILE_OPS 61.9%→95.2%, COMPREHENSION 0%→33.3%, but CODING 85.7%→0%)

## Experiment 13 — discard
**eval_loss:** 1.0143 | **checklist:** 0.33 | **Change:** Added hermes-agent-reasoning-traces dataset (+4047 rows, total 20437→5404 after filter)
**Reasoning:** More diverse tool-use reasoning traces might improve generalization
**Result:** eval_loss improved marginally (1.017→1.014) but checklist still 3/9. Prompts 2-3 still produce prose-style reasoning without think tags. Hermes data likely dilutes think-tag format learning. Also caused 2 thermal shutdowns before adding throttle. Reverted dataset addition, kept safety fixes (thermal throttle, OOM guards, GGUF export).

## Experiment 14 — keep
**eval_loss:** 0.9850 | **checklist:** 0.67 | **Change:** MAX_STEPS 600→700 (with grad_accum=8)
**Reasoning:** 600 steps learns eval_loss well but not think-tag format for non-math prompts. exp4 showed 900 steps gets 9/9 checklist. exp6 tried 700 with grad_accum=4 and got checklist 0.67. With grad_accum=8, 700 steps may pass quality gate without overfitting.
**Result:** Best results yet! First eval_loss below 1.0 (1.017→0.985). Checklist 6/9 (67%) — passes quality gate! Prompts 1 and 3 pass all checks, Prompt 2 (sky explanation) still fails think tags. Major milestone.
**PinchBench:** 66.7% score | 3,314,983 tokens | 72 requests (CODING 0%→100% recovered, but CALENDAR 83.3%→0%, FILE_OPS 95.2%→66.7%. Token usage 3x higher — model more verbose/loopy at 700 steps)

## Experiment 15 — discard
**eval_loss:** 14.3030 | **checklist:** 0.67 | **Change:** LORA_ALPHA 16→32 (2x rank)
**Result:** Catastrophic. Alpha must equal rank for Qwen3.5. Reverted.

## Experiment 16 — discard
**eval_loss:** 0.9909 | **checklist:** 0.33 | **Change:** WEIGHT_DECAY 0.01→0.0
**Result:** eval_loss regressed, checklist dropped. Weight decay 0.01 is beneficial. Reverted.

## Experiment 17 — discard
**eval_loss:** 0.9995 | **checklist:** 0.00 | **Change:** WARMUP_STEPS 10→50
**Result:** Both metrics worse. Too much warmup wastes steps on 700-step budget. Reverted.

## Experiment 18 — discard
**eval_loss:** 11.9478 | **checklist:** 0.67 | **Change:** GRADIENT_ACCUMULATION 8→16
**Result:** Catastrophic. Only ~44 optimizer updates — far too few. Reverted.

## Experiment 19 — discard
**eval_loss:** 1.0134 | **checklist:** 0.67 | **Change:** WEIGHT_DECAY 0.01→0.1
**Result:** Too much regularization constrains learning. Reverted.

## Experiment 20 — discard
**eval_loss:** 0.9910 | **checklist:** 0.67 | **Change:** OPTIMIZER adamw_8bit→adamw_torch
**Result:** eval_loss regressed. 8-bit optimizer is sufficient. Switched to GPU clock lock (2000MHz) for thermal management. Reverted.

## Experiment 21 — discard
**eval_loss:** 0.9841 | **checklist:** 0.33 | **Change:** LEARNING_RATE 1e-4→1.5e-4
**Result:** New best eval_loss (0.984) but checklist crashed to 33% — quality gate failed. Reverted.

## Experiment 22 — discard
**eval_loss:** NaN | **checklist:** 0.33 | **Change:** MAX_STEPS 700→750
**Result:** NaN eval_loss — confirmed 750 steps is NaN boundary (same as exp5). Reverted.

## Experiment 23 — discard
**eval_loss:** NaN | **checklist:** 0.67 | **Change:** EVAL_SPLIT 0.05→0.10
**Result:** NaN eval_loss. Larger eval split reduced training data, didn't help. Reverted.

## Experiment 24 — keep (9B baseline)
**eval_loss:** 0.9076 | **checklist:** 0.00 | **Change:** MODEL_NAME Qwen3.5-4B→Qwen3.5-9B
**Reasoning:** 4B config exhausted after 23 experiments (exp15-23 all discards). 9B has more capacity for reasoning. 128GB unified memory can handle it.
**Result:** Massive eval_loss improvement (0.985→0.908). But checklist 0/9 — model uses prose "Thinking Process:" format instead of <think> tags. Same pattern as early 4B experiments. VRAM: 30GB (vs 21GB for 4B). Training time: 166 min (vs ~108 min). Fixed meta tensor offloading bug for 9B compatibility.

