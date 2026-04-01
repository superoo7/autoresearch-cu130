# autoresearch-cu130

Autonomous SFT finetuning research for Qwen3.5-4B on DGX Spark (NVIDIA GB10, CUDA 13.0).

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted from pre-training to **finetuning** with [Unsloth](https://github.com/unslothai/unsloth) + LoRA.

## What this does

An AI agent autonomously finetunes [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) to improve its reasoning ability. It modifies `train.py` (LoRA config, hyperparameters, dataset filtering), runs training, evaluates the result using **two metrics**, keeps improvements or discards failures, and repeats — all without human intervention.

Training uses **completion-only masking** — loss is only computed on the assistant response (from `<|im_start|>assistant` onward), so the model focuses on learning reasoning + answer generation, not predicting user prompts.

- **`prepare.py`** — one-time dataset download and preparation. Protected from agent modification.
- **`train.py`** — the single file the agent edits. All config, training, evaluation, and logging.
- **`program.md`** — instructions for the agent. Edited by humans.

## How evaluation works

**Layer 1: eval_loss** — cross-entropy on a 5% held-out eval split. Lower is better. Fast, quantitative.

**Layer 2: checklist scoring** — after training, the model generates answers to test prompts. Each output is scored against binary yes/no checks:

| Check | What it catches |
|---|---|
| Uses `<think>` tags? | Models that skip chain-of-thought |
| Step-by-step reasoning inside thinking? | Shallow reasoning that just restates the question |
| Clear answer outside `</think>` tags? | Models that bury the answer in reasoning |

The agent keeps a change only if eval_loss improved **AND** checklist score stays above 60%.

## Dataset

16,390 high-quality reasoning distillation examples from 5 sources:

| Dataset | Rows | Purpose |
|---|---|---|
| [nohurry/Opus-4.6-Reasoning-3000x-filtered](https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered) | 2,326 | Claude 4.6 Opus reasoning trajectories |
| [TeichAI/claude-4.5-opus-high-reasoning-250x](https://huggingface.co/datasets/TeichAI/claude-4.5-opus-high-reasoning-250x) | 250 | High-intensity structured reasoning instances |
| [Jackrong/Qwen3.5-reasoning-700x](https://huggingface.co/datasets/Jackrong/Qwen3.5-reasoning-700x) | 633 | Step-by-step problem solving and reasoning diversity |
| [Roman1111111/gemini-3.1-pro-hard-high-reasoning](https://huggingface.co/datasets/Roman1111111/gemini-3.1-pro-hard-high-reasoning) | 3,150 | Gemini 3.1 Pro hard/high reasoning across domains |
| [Roman1111111/gemini-3-pro-10000x-hard-high-reasoning](https://huggingface.co/datasets/Roman1111111/gemini-3-pro-10000x-hard-high-reasoning) | 10,031 | Large-scale Gemini 3 Pro hard reasoning (astrophysics, math, CS, etc.) |

All datasets are unified into a single chat format with `<think>...</think>` tags for chain-of-thought reasoning by `prepare.py`.

## Quick start

```bash
# Start Docker container
docker run -it --gpus=all --net=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):$(pwd) -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w $(pwd) unsloth-dgx-spark

# Prepare data (one-time, downloads from HuggingFace)
python prepare.py

# Run a single training experiment
python train.py 2>&1 | tee run.log

# Check results
grep "^eval_loss:\|^checklist_score:" run.log
```

## Running the agent

Spin up Claude Code (or similar) in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment!
```

The agent will establish a baseline, then loop autonomously: modify config, train, evaluate, keep/revert, repeat. See `program.md` for the full protocol.

## Project structure

```
prepare.py      — dataset download + preparation (do not modify)
train.py        — LoRA config, training, evaluation (agent modifies this)
program.md      — agent instructions + experiment loop protocol
references/     — eval-guide.md for writing good binary evals
Dockerfile      — Docker image build (unsloth-dgx-spark)
```

## Hardware

NVIDIA DGX Spark: GB10 GPU, 128GB unified memory, CUDA 13.0.

## License

MIT
