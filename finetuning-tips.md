# Fine-Tuning Tips: Qwen3.5 with LoRA on Unsloth

Compiled from 14 experiments on Qwen3.5-4B (DGX Spark, 128GB unified memory) and Unsloth documentation.

## Qwen3.5-Specific

- **Use transformers v5** — older versions will not work with Qwen3.5.
- **Do NOT use QLoRA (4-bit)** — BitsAndBytes has quantization issues with Qwen3.5. Use bf16 LoRA instead (`load_in_16bit=True`).
- **MoE models (32B-A3B)**: bf16 strongly preferred over QLoRA. VRAM ~74GB for 35B-A3B.
- **Kernel compilation is slow** — Qwen3.5 uses custom Mamba Triton kernels. First run compiles them; subsequent runs are faster.
- **Chat template auto-injects `<think>` tags** — if your dataset already has them, strip them to avoid double tags.
- **Reasoning dataset mix**: keep at least 75% reasoning examples if you want to preserve thinking ability.

## LoRA Configuration

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| rank | 16 | 32 hurt eval_loss (exp2). 8 lacked capacity (exp9). 16 is the sweet spot. |
| alpha | 16 (= rank) | Unsloth docs recommend alpha = rank for Qwen3.5. |
| target_modules | All 7 (q/k/v/o + gate/up/down) | Attention-only was significantly worse (exp11: +0.05 eval_loss). MLP projections are essential. |
| dropout | 0 | Standard for LoRA. |
| gradient_checkpointing | "unsloth" | Reduces VRAM and extends context capacity. |

## Hyperparameters That Worked

**Best config (eval_loss 0.985):** LR=1e-4, cosine schedule, 700 steps, grad_accum=8, rank=16, seq=2048, ~5K rows.

| Parameter | Best Value | What We Tried | Learnings |
|-----------|-----------|---------------|-----------|
| Learning rate | 1e-4 | 5e-5, 1e-4, 2e-4 | 2e-4 was too aggressive; 5e-5 underfit at 600 steps. 1e-4 was the sweet spot. For larger models, start lower (3e-5 to 5e-5). |
| LR scheduler | cosine | cosine, linear | Cosine beat linear (exp7). |
| Max steps | 700 | 300, 600, 700, 750, 900 | 300 too few. 700 with grad_accum=8 gave best eval_loss (0.985) and first checklist pass (67%). 700 with grad_accum=4 overfits. 900 overfits regardless. |
| Grad accumulation | 8 | 4, 8 | 8 gave best eval_loss ever — larger effective batch stabilizes gradients. |
| Batch size | 1 | 1 | Keep at 1, increase grad_accum instead to simulate larger batches without VRAM cost. |
| Sequence length | 2048 | 2048, 4096 | 4096 tripled the dataset (more rows pass filter) and eval_loss exploded at 600 steps. If increasing seq length, must also increase steps proportionally. |
| Optimizer | adamw_8bit | adamw_8bit | Saves VRAM vs adamw_torch. |
| Weight decay | 0.01 | 0.01 | Standard value, not extensively tuned. |

## Dataset

- **5K rows at 600 steps** is a good balance for rapid experimentation (~30-50 min/run).
- **Quality > quantity** — the LIMA paper showed 1K curated examples can rival 50K+ low-quality ones.
- **Dataset size must match step budget** — 15K rows at 600 steps was catastrophic (exp10: eval_loss 12.6). Rule of thumb: ~1 epoch coverage minimum.
- **Set a MAX_DATASET_ROWS safety cap** to prevent OOM from dataset growth.
- **Completion-only training** (`completion_only_loss=True`) focuses learning on assistant responses, not user/system tokens.

## Memory Management (DGX Spark / Unified Memory)

- **Free trainer before GGUF export** — `del trainer; torch.cuda.empty_cache(); gc.collect()`. GGUF quantization loads the full model again; combined with training state this can exceed 128GB.
- **OOM triggers system shutdown** on unified memory systems — there's no graceful fallback.
- **Thermal shutdown** — DGX Spark GB10 hits 85°C+ under sustained training. Use `THROTTLE_SECONDS = 1.0` (1s sleep between steps) to prevent thermal shutdown. GPU temp should stay below 90°C. Monitor with `nvidia-smi`.
- **Peak VRAM for 4B bf16 + LoRA**: ~21GB. GGUF export adds significant overhead on top.
- **If OOM**: reduce `MAX_SEQ_LENGTH` or `BATCH_SIZE` first.

## Overfitting Signals

- **Train loss drops but eval loss rises** — classic overfitting. Happened at 900 steps (exp4).
- **Checklist score improves while eval_loss worsens** — the model is memorizing output format but losing generalization (exp4: perfect checklist, bad eval_loss).
- **Grad accumulation matters for overfitting threshold** — 700 steps with grad_accum=4 overfits (exp6), but 700 steps with grad_accum=8 is the sweet spot (exp14: best eval_loss 0.985).
- **PinchBench can diverge from eval_loss** — exp14 had best eval_loss but PinchBench regressed (66.7% vs 78.0%). More steps can make the model verbose/loopy in agent tasks even while improving loss.
- **More epochs is risky** — with small datasets, >1 epoch likely overfits.

## Evaluation

- **eval_loss is the primary metric** — lower = better reasoning ability.
- **Checklist scoring** catches format issues (think tags, step-by-step, answer structure) but can be misleading — perfect checklist with bad eval_loss means overfitting.
- **PinchBench** (real-world agent benchmark, 9 automated tasks) is directional only — a single task flip swings score ~11%. Best: exp12 at 78.0% (FILE_OPS 95.2%). Exp14 had best eval_loss but PinchBench regressed to 66.7% (verbose/loopy behavior).
- **eval_loss and PinchBench can conflict** — lower eval_loss doesn't always mean better agent performance. The model may learn reasoning format but become verbose, hurting task completion.
- **Always test with manual inference** — metrics don't catch everything.

## Scaling to Larger Models

When moving from 4B to Qwen3.5-32B-A3B:
- **Lower the learning rate** — start at 3e-5 to 5e-5 instead of 1e-4.
- **LoRA rank may need to scale** — but test rank 16 first since active params (3B) are similar to 4B.
- **Can use the full 20K dataset** — more model capacity + more steps to absorb it.
- **Relative rankings transfer** — which LR schedule, rank, and dataset filtering works best tends to hold across model sizes.
- **bf16 is mandatory for MoE** — do not attempt QLoRA on MoE models.

## Common Mistakes

1. **Changing multiple hyperparams at once** — you won't know what helped. Change ONE thing per experiment.
2. **Jumping to full fine-tuning** — if LoRA fails, FFT won't magically fix it. Fix your data/config first.
3. **Ignoring the dataset/steps ratio** — more data without more steps = undertrained model.
4. **Not freeing memory before export** — GGUF export on top of training state causes OOM.
5. **Using 4-bit QLoRA on Qwen3.5** — known BitsAndBytes issues. Use bf16 instead.
