"""
Autoresearch finetuning script. Single-GPU, single-file.
SFT finetune with LoRA using Unsloth on DGX Spark.
Usage: python train.py
"""

# ============================================================
# CONFIG — Agent modifies this section
# ============================================================

# Model
MODEL_NAME = "unsloth/Qwen3.5-9B"        # Upgraded from 4B to 9B for more capacity
MAX_SEQ_LENGTH = 2048                     # Max tokens per example. 1024/2048/4096/8192
LOAD_IN_4BIT = False                      # DO NOT use 4-bit QLoRA on Qwen3.5 (BitsandBytes issue)
LOAD_IN_16BIT = True                      # Use bf16 instead (~10GB VRAM for 4B model)

# LoRA
LORA_RANK = 16                            # LoRA rank. Higher = more capacity. Try 4/8/16/32/64
LORA_ALPHA = 16                           # Scaling factor. Unsloth docs recommend alpha = rank for Qwen3.5
LORA_TARGET_MODULES = [                   # Layers to apply LoRA to
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Dataset
DATASET_PATH = "datasets/combined_reasoning"
FILTER_BY_LENGTH = True                   # Drop rows exceeding MAX_SEQ_LENGTH
MAX_DATASET_ROWS = 10000                  # Safety cap: limit rows to avoid OOM. None = no limit
EVAL_SPLIT = 0.05                         # 5% for eval (higher = more stable metric)

# Training
BATCH_SIZE = 1                            # Per-device batch size
GRADIENT_ACCUMULATION = 8                 # Effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION
LEARNING_RATE = 1e-4                      # Peak LR. Try 5e-5 to 5e-4
NUM_EPOCHS = 1                            # Passes over dataset. Watch for overfitting >3
MAX_STEPS = 900                           # More steps for 9B to learn think-tag format (4B needed 700+)
WARMUP_STEPS = 10                         # LR warmup steps
LR_SCHEDULER = "cosine"                   # "cosine", "linear", "constant"
OPTIMIZER = "adamw_8bit"                  # adamw_8bit saves VRAM vs adamw_torch
WEIGHT_DECAY = 0.01                       # L2 regularization
LOGGING_STEPS = 10                        # Log every N steps
EVAL_STEPS = 100                          # Evaluate every N steps
SAVE_STEPS = 300                          # Checkpoint at end of run
OUTPUT_DIR = "outputs"                    # Checkpoints and adapters
RESUME_FROM_CHECKPOINT = None             # Set to checkpoint path (e.g. "outputs/checkpoint-300") to resume

# HuggingFace Hub upload
HF_UPLOAD = False                            # Set True to upload LoRA + GGUF after training
HF_REPO_ID = "superoo7/autoresearch-qwen3.5" # HuggingFace repo (run `huggingface-cli login` first)

# Thermal throttle — pause between steps to prevent DGX Spark thermal shutdown
THROTTLE_SECONDS = 1.0                    # 1s throttle + 2000MHz clock lock for 9B stability

# Completion-only training — only compute loss on assistant response, not user/system tokens
COMPLETION_ONLY = True                    # True = train only on assistant response

# Inference test prompts
TEST_PROMPTS = [
    "Solve step by step: What is 23 * 47 + 15?",
    "Explain why the sky is blue using physics.",
    "Write a Python function to check if a string is a palindrome.",
]
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

# Output quality checklist — binary yes/no checks on generated outputs
# These act as a quality gate: keep a change only if checklist >= CHECKLIST_PASS_THRESHOLD
CHECKLIST = [
    {
        "name": "uses_think_tags",
        "question": "Does the response contain <think> and </think> tags?",
        "check": lambda text: "<think>" in text and "</think>" in text,
    },
    {
        "name": "step_by_step_reasoning",
        "question": "Does the thinking section show step-by-step breakdown?",
        "check": lambda text: (
            "<think>" in text and
            any(marker in text.lower() for marker in
                ["step ", "first", "then", "next", "finally", "1.", "2.", "1)", "2)"])
        ),
    },
    {
        "name": "answer_outside_tags",
        "question": "Is there a clear answer after the </think> closing tag?",
        "check": lambda text: (
            "</think>" in text and
            len(text.split("</think>", 1)[-1].strip()) > 10
        ),
    },
]
CHECKLIST_PASS_THRESHOLD = 0.6            # Quality gate: revert if below this

# ============================================================
# IMPORTS + SETUP
# ============================================================

import math
import os
import time
import csv
from datetime import datetime

import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

steps_label = f"s{MAX_STEPS}" if MAX_STEPS > 0 else f"ep{NUM_EPOCHS}"
run_name = f"r{LORA_RANK}_a{LORA_ALPHA}_lr{LEARNING_RATE}_{steps_label}_seq{MAX_SEQ_LENGTH}"
start_time = time.time()

print(f"=== RUN: {run_name} ===")
print(f"torch: {torch.__version__} | cuda: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}")
print()

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_16bit=LOAD_IN_16BIT,
    full_finetuning=False,
)

# Unsloth 2026.3+ returns a Qwen3VLProcessor for Qwen3.5 models.
# Extract the underlying text tokenizer for encode/decode operations.
text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=MAX_SEQ_LENGTH,
)

# Count trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
print()

# ============================================================
# LOAD & PREPARE DATASET
# ============================================================

print("Loading dataset...")
dataset = load_from_disk(DATASET_PATH)
print(f"Loaded: {len(dataset)} rows")

# Format messages to text
# Note: Qwen3/3.5 chat templates auto-insert <think> tags in the assistant response.
# Our dataset already has <think>...</think> in the assistant content, so we strip them
# to avoid double <think> tags. The template will re-add them.
def strip_think_tags(content):
    """Remove <think>...</think> wrapper, keep the inner reasoning + answer."""
    import re
    # Match <think>...reasoning...</think>\n\nanswer
    match = re.match(r"<think>\s*(.*?)\s*</think>\s*(.*)", content, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = match.group(2).strip()
        # Return thinking + answer without tags (template will wrap in <think>)
        return thinking + "\n\n" + answer if answer else thinking
    return content

def to_text(example):
    msgs = []
    for m in example["messages"]:
        if m["role"] == "assistant":
            msgs.append({"role": "assistant", "content": strip_think_tags(m["content"])})
        else:
            msgs.append(m)
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

dataset = dataset.map(to_text, num_proc=1)

# Filter by length
if FILTER_BY_LENGTH:
    before = len(dataset)
    dataset = dataset.filter(lambda x: len(text_tokenizer.encode(x["text"])) <= MAX_SEQ_LENGTH)
    print(f"Filtered: {before} -> {len(dataset)} rows (dropped {before - len(dataset)})")

# Safety cap on dataset size to prevent OOM
if MAX_DATASET_ROWS and len(dataset) > MAX_DATASET_ROWS:
    dataset = dataset.shuffle(seed=42).select(range(MAX_DATASET_ROWS))
    print(f"Capped to {MAX_DATASET_ROWS} rows (safety limit)")

# Train/eval split
split = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
print()

# ============================================================
# TRAIN
# ============================================================

class ThermalThrottleCallback(TrainerCallback):
    """Sleep between training steps to prevent thermal shutdown."""
    def on_step_end(self, args, state, control, **kwargs):
        if THROTTLE_SECONDS > 0:
            time.sleep(THROTTLE_SECONDS)

# Fix meta tensors — 9B model may offload lm_head to meta device
# Materialize it before SFTTrainer's fix_untrained_tokens runs
for name, param in model.named_parameters():
    if param.device.type == "meta":
        print(f"Materializing meta tensor: {name} ({param.shape})")
        param.data = torch.zeros(param.shape, dtype=param.dtype, device="cuda")

print("Setting up trainer...")

if COMPLETION_ONLY:
    print(f"Completion-only training enabled via SFTConfig")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        completion_only_loss=COMPLETION_ONLY,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        max_seq_length=MAX_SEQ_LENGTH,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=3407,
        dataset_num_proc=1,
        report_to="none",
        output_dir=OUTPUT_DIR,
    ),
    callbacks=[ThermalThrottleCallback()] if THROTTLE_SECONDS > 0 else [],
)

print("Training...")
train_result = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
train_loss = train_result.training_loss
print()

# ============================================================
# EVALUATE
# ============================================================

print("Evaluating...")
eval_results = trainer.evaluate()
eval_loss = float(eval_results["eval_loss"])
perplexity = math.exp(eval_loss)

peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
train_time = time.time() - start_time

print()

# ============================================================
# INFERENCE TEST
# ============================================================

print("=" * 60)
print("INFERENCE TEST + CHECKLIST SCORING")
print("=" * 60)

responses = []
for i, prompt in enumerate(TEST_PROMPTS):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = text_tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
        )
    response = text_tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    # The chat template injects <think> in the prompt, so it's not in the generated response.
    # Prepend it so checklist scoring can detect the full <think>...</think> structure.
    if "</think>" in response and "<think>" not in response:
        response = "<think>" + response
    responses.append(response)
    print(f"\n--- Prompt {i+1}: {prompt}")
    print(f"--- Response:\n{response[:1000]}")
    if len(response) > 1000:
        print(f"... (truncated, {len(response)} chars total)")

# Score outputs against checklist
print("\n" + "=" * 60)
print("CHECKLIST RESULTS")
print("=" * 60)

total_checks = 0
total_passed = 0
for i, response in enumerate(responses):
    print(f"\nPrompt {i+1}:")
    for item in CHECKLIST:
        passed = item["check"](response)
        total_checks += 1
        total_passed += int(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {item['name']}: {item['question']}")

checklist_score = total_passed / total_checks if total_checks > 0 else 0.0
print(f"\nChecklist: {total_passed}/{total_checks} ({checklist_score:.0%})")
if checklist_score < CHECKLIST_PASS_THRESHOLD:
    print(f"WARNING: Below quality gate threshold ({CHECKLIST_PASS_THRESHOLD:.0%})")
print()

# ============================================================
# RESULTS SUMMARY + LOG
# ============================================================

# Print parseable summary (like autoresearch)
print("---")
print(f"eval_loss:        {eval_loss:.6f}")
print(f"perplexity:       {perplexity:.4f}")
print(f"train_loss:       {train_loss:.6f}")
print(f"checklist_score:  {checklist_score:.2f}")
print(f"checklist_detail: {total_passed}/{total_checks}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"train_time_min:   {train_time/60:.1f}")
print(f"dataset_rows:     {len(train_dataset)}")
print(f"lora_rank:        {LORA_RANK}")
print(f"lora_alpha:       {LORA_ALPHA}")
print(f"learning_rate:    {LEARNING_RATE}")
print(f"max_steps:        {MAX_STEPS}")
print(f"num_epochs:       {NUM_EPOCHS}")
print(f"seq_length:       {MAX_SEQ_LENGTH}")
print(f"batch_size:       {BATCH_SIZE}")
print(f"grad_accum:       {GRADIENT_ACCUMULATION}")
print(f"completion_only:  {COMPLETION_ONLY}")

# Append to results.tsv
results_file = "results.tsv"
header = ["timestamp", "run_name", "eval_loss", "perplexity", "train_loss",
          "checklist_score", "peak_vram_mb", "train_time_min", "lora_rank",
          "lora_alpha", "learning_rate", "num_epochs", "seq_length",
          "batch_size", "grad_accum", "dataset_rows"]

write_header = not os.path.exists(results_file)
with open(results_file, "a", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    if write_header:
        writer.writerow(header)
    writer.writerow([
        datetime.now().isoformat(timespec="seconds"),
        run_name,
        f"{eval_loss:.6f}",
        f"{perplexity:.4f}",
        f"{train_loss:.6f}",
        f"{checklist_score:.2f}",
        f"{peak_vram_mb:.1f}",
        f"{train_time/60:.1f}",
        LORA_RANK,
        LORA_ALPHA,
        LEARNING_RATE,
        NUM_EPOCHS,
        MAX_SEQ_LENGTH,
        BATCH_SIZE,
        GRADIENT_ACCUMULATION,
        len(train_dataset),
    ])
print(f"\nResults appended to {results_file}")

# ============================================================
# SAVE
# ============================================================

adapter_dir = f"{OUTPUT_DIR}/lora_adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"LoRA adapter saved to {adapter_dir}")

# ============================================================
# EXPORT GGUF (for ollama / PinchBench)
# ============================================================

# Free training state before GGUF export to avoid OOM on unified memory systems
del trainer
torch.cuda.empty_cache()
import gc; gc.collect()
print(f"Freed trainer memory. VRAM after cleanup: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

gguf_dir = f"{OUTPUT_DIR}/gguf-{run_name}"
print(f"\nExporting GGUF to {gguf_dir}...")
try:
    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"gguf_path: {gguf_dir}")
except Exception as e:
    print(f"GGUF export failed: {e}")

# ============================================================
# UPLOAD TO HUGGINGFACE HUB
# ============================================================

if HF_UPLOAD:
    from huggingface_hub import HfApi
    print(f"\nUploading to HuggingFace Hub: {HF_REPO_ID}")
    api = HfApi()

    # Upload LoRA adapter
    print(f"Uploading LoRA adapter from {adapter_dir}...")
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=HF_REPO_ID,
        path_in_repo="lora_adapter",
        repo_type="model",
        create_pr=False,
        commit_message=f"Upload LoRA adapter ({run_name})",
    )

    # Upload GGUF
    if os.path.isdir(gguf_dir):
        print(f"Uploading GGUF from {gguf_dir}...")
        api.upload_folder(
            folder_path=gguf_dir,
            repo_id=HF_REPO_ID,
            path_in_repo="gguf",
            repo_type="model",
            create_pr=False,
            commit_message=f"Upload GGUF q4_k_m ({run_name})",
        )

    print(f"Uploaded to https://huggingface.co/{HF_REPO_ID}")

total_time = time.time() - start_time
print(f"\nDone in {total_time/60:.1f} minutes.")
