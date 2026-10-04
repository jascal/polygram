"""CPU-friendly host competence screening; no representation fitting."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import subprocess
import time

from .data import SYSTEM, generate, read_dataset, write_dataset


def load_host(model_id, threads=4, dtype="float32"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, dtype=getattr(torch, dtype),
        attn_implementation="eager",
    ).eval()
    model.requires_grad_(False)
    return model, tokenizer


def tokenize_prompt(tokenizer, ex, unrelated=False):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": ex.prompt(unrelated)}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["split"])].append(row)
    out = {}
    for (task, split), values in groups.items():
        pairs = defaultdict(list)
        for row in values:
            pairs[row["pair_id"]].append(row["correct"])
        accuracy = sum(row["correct"] for row in values) / len(values)
        pair_accuracy = sum(all(p) and len(p) == 2 for p in pairs.values()) / len(pairs)
        out[f"{task}/{split}"] = {
            "n": len(values), "pairs": len(pairs), "accuracy": accuracy,
            "pair_accuracy": pair_accuracy,
            "unrestricted_first_token_accuracy": sum(r["first_token_correct"] for r in values)
            / len(values),
            "mean_margin_nats": sum(r["margin_nats"] for r in values) / len(values),
            "competence_pass": accuracy >= 0.90 and pair_accuracy >= 0.80,
        }
    return out


def score_examples(model, tokenizer, examples, unrelated=False, progress=True):
    import torch

    rows = []
    with torch.inference_mode():
        for index, ex in enumerate(examples):
            ids = tokenize_prompt(tokenizer, ex, unrelated)
            # transformers 5 may return a BatchEncoding from apply_chat_template.
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            token_ids = torch.tensor([ids])
            logits = model(input_ids=token_ids, use_cache=False,
                           logits_to_keep=1).logits[0, -1].float()
            log_probs = logits.log_softmax(-1)
            answer = ex.unrelated_answer if unrelated else ex.answer
            scores, first_ids = {}, {}
            for candidate in ex.candidates:
                continuation = tokenizer.encode(candidate, add_special_tokens=False)
                first_ids[candidate] = continuation[0]
                score = float(log_probs[continuation[0]])
                if len(continuation) > 1:
                    full = torch.tensor([ids + continuation[:-1]])
                    lp = model(input_ids=full, use_cache=False).logits[0].float().log_softmax(-1)
                    score += sum(float(lp[len(ids) + j - 1, continuation[j]])
                                 for j in range(1, len(continuation)))
                scores[candidate] = score
            prediction = max(scores, key=scores.get)
            rows.append({
                "id": ex.id, "pair_id": ex.pair_id, "task": ex.task, "split": ex.split,
                "answer": answer, "prediction": prediction, "correct": prediction == answer,
                "first_token_correct": int(logits.argmax()) == first_ids[answer],
                "argmax_id": int(logits.argmax()), "scores": scores,
                "margin_nats": scores[answer] - max(v for k, v in scores.items() if k != answer),
                "prompt_tokens": len(ids), "unrelated": unrelated,
            })
            if progress and (index + 1) % 12 == 0:
                print(f"scored {index + 1}/{len(examples)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--unrelated", action="store_true")
    args = parser.parse_args()
    started = time.time()
    examples = read_dataset(args.dataset) if args.dataset else generate()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = write_dataset(args.output / "dataset.jsonl", examples)
    dev = [ex for ex in examples if ex.split == "dev"]
    model, tokenizer = load_host(args.model, args.threads, args.dtype)
    rows = score_examples(model, tokenizer, dev, args.unrelated)
    import torch
    import transformers

    report = {
        "model": args.model, "revision": getattr(model.config, "_commit_hash", None),
        "dataset": manifest, "split": "dev", "unrelated": args.unrelated,
        "device": "cpu", "dtype": args.dtype, "threads": args.threads,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "polygram_rev": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "elapsed_seconds": time.time() - started, "summary": summarize(rows),
    }
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
