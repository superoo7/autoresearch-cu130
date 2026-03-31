# autoresearch-cu130

Autonomous SFT finetuning research for GPT-OSS 20B on DGX Spark (NVIDIA GB10, CUDA 13.0).

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted from pre-training to finetuning with [Unsloth](https://github.com/unslothai/unsloth) + LoRA.

## How it works

An AI agent modifies `train.py` (LoRA config, hyperparameters, dataset filtering), runs finetuning, checks `eval_loss`, keeps improvements or discards failures, and repeats autonomously.

- **`prepare.py`** — one-time dataset download and preparation. Not modified by agents.
- **`train.py`** — the single file the agent edits. Config, training, evaluation, logging.
- **`program.md`** — instructions for the agent. Edited by humans.

The metric is **eval_loss** (cross-entropy on held-out eval split) — lower is better.

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
```

## Running the agent

Spin up Claude Code or similar in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment!
```

## Dataset

3,209 curated reasoning examples from 3 sources:

| Source | Rows | Description |
|---|---|---|
| Opus-4.6-Reasoning-3000x-filtered | 2,326 | Math/code reasoning with chain-of-thought |
| claude-4.5-opus-high-reasoning-250x | 250 | High-quality coding with deep reasoning |
| Qwen3.5-reasoning-700x | 633 | Math reasoning traces |

## Project structure

```
prepare.py      — dataset download + preparation (do not modify)
train.py        — LoRA config, training, evaluation (agent modifies this)
program.md      — agent instructions
Dockerfile      — Docker image build
```

## Hardware

NVIDIA DGX Spark: GB10 GPU, 128GB unified memory, CUDA 13.0.

## License

MIT
