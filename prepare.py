# PROTECTED: Do not modify. Agent must not edit this file.
# Run once to download and prepare datasets.
# Usage: python prepare.py
"""
Prepare and unify 3 reasoning datasets into a single training-ready dataset.

Input datasets (downloaded from HuggingFace):
  - nohurry/Opus-4.6-Reasoning-3000x-filtered (2326 rows)
  - TeichAI/claude-4.5-opus-high-reasoning-250x (250 rows)
  - Jackrong/Qwen3.5-reasoning-700x (633 rows)

Output:
  - datasets/combined_reasoning/ : unified dataset with "messages" column
    Each row: [{"role": "user", ...}, {"role": "assistant", ...}]
    Assistant content uses <think>...</think> tags for reasoning.
"""

import subprocess
from datasets import load_dataset, Dataset, concatenate_datasets
from pathlib import Path

BASE = Path("datasets")
OUTPUT = BASE / "combined_reasoning"

HF_DATASETS = {
    "Opus-4.6-Reasoning-3000x-filtered": "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "claude-4.5-opus-high-reasoning-250x": "TeichAI/claude-4.5-opus-high-reasoning-250x",
    "Qwen3.5-reasoning-700x": "Jackrong/Qwen3.5-reasoning-700x",
    "gemini-3.1-pro-hard-high-reasoning": "Roman1111111/gemini-3.1-pro-hard-high-reasoning",
    "gemini-3-pro-10000x-hard-high-reasoning": "Roman1111111/gemini-3-pro-10000x-hard-high-reasoning",
}


def download_datasets():
    """Download datasets from HuggingFace if not already present."""
    BASE.mkdir(exist_ok=True)
    for dirname, repo_id in HF_DATASETS.items():
        local_dir = BASE / dirname
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"  {dirname}: already exists, skipping download")
            continue
        print(f"  {dirname}: downloading from {repo_id}...")
        subprocess.run(
            ["hf", "download", "--repo-type", "dataset", repo_id, "--local-dir", str(local_dir)],
            check=True,
        )
    print()


def load_opus():
    """Opus-4.6: separate thinking + solution columns -> wrap thinking in <think> tags."""
    ds = load_dataset(str(BASE / "Opus-4.6-Reasoning-3000x-filtered"))["train"]
    rows = []
    for r in ds:
        assistant_content = f"<think>\n{r['thinking']}\n</think>\n\n{r['solution']}"
        rows.append({
            "messages": [
                {"role": "user", "content": r["problem"]},
                {"role": "assistant", "content": assistant_content},
            ],
            "source": "opus-4.6",
            "category": r.get("category", ""),
            "difficulty": r.get("difficulty", ""),
        })
    print(f"  Opus-4.6: {len(rows)} rows")
    return Dataset.from_list(rows)


def load_claude():
    """Claude-4.5-Opus: already chat format, drop empty system message."""
    ds = load_dataset(str(BASE / "claude-4.5-opus-high-reasoning-250x"))["train"]
    rows = []
    for r in ds:
        msgs = r["messages"]
        filtered = [m for m in msgs if not (m["role"] == "system" and m["content"].strip() == "")]
        rows.append({
            "messages": filtered,
            "source": "claude-4.5-opus",
            "category": "",
            "difficulty": "",
        })
    print(f"  Claude-4.5-Opus: {len(rows)} rows")
    return Dataset.from_list(rows)


def load_qwen():
    """Qwen3.5: convert human/gpt roles to user/assistant."""
    ds = load_dataset(str(BASE / "Qwen3.5-reasoning-700x"))["train"]
    role_map = {"human": "user", "gpt": "assistant"}
    rows = []
    for r in ds:
        msgs = [{"role": role_map[m["from"]], "content": m["value"]} for m in r["conversation"]]
        rows.append({
            "messages": msgs,
            "source": "qwen3.5",
            "category": r.get("domain", ""),
            "difficulty": "",
        })
    print(f"  Qwen3.5: {len(rows)} rows")
    return Dataset.from_list(rows)


def load_gemini(dirname):
    """Gemini datasets: original_data/original_input (dict) + model_thoughts + model_response."""
    ds = load_dataset(str(BASE / dirname))["train"]
    rows = []
    for r in ds:
        # Both datasets use slightly different key names
        orig = r.get("original_data") or r.get("original_input") or {}
        problem = orig.get("text", "")
        thinking = r.get("model_thoughts", "")
        solution = r.get("model_response", "")
        assistant_content = f"<think>\n{thinking}\n</think>\n\n{solution}"
        rows.append({
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": assistant_content},
            ],
            "source": dirname,
            "category": orig.get("domain", ""),
            "difficulty": orig.get("difficulty", ""),
        })
    print(f"  {dirname}: {len(rows)} rows")
    return Dataset.from_list(rows)


def main():
    print("Step 1: Downloading datasets...")
    download_datasets()

    print("Step 2: Loading and transforming...")
    ds_opus = load_opus()
    ds_claude = load_claude()
    ds_qwen = load_qwen()
    ds_gemini31 = load_gemini("gemini-3.1-pro-hard-high-reasoning")
    ds_gemini3 = load_gemini("gemini-3-pro-10000x-hard-high-reasoning")

    combined = concatenate_datasets([ds_opus, ds_claude, ds_qwen, ds_gemini31, ds_gemini3])
    combined = combined.shuffle(seed=42)

    print(f"\nCombined: {len(combined)} rows")
    sources = {}
    for r in combined:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    combined.save_to_disk(str(OUTPUT))
    print(f"\nDataset ready at: {OUTPUT}")


if __name__ == "__main__":
    main()
