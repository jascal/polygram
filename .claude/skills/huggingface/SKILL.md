---
name: huggingface
description: HuggingFace Hub operations — download models/datasets/files, upload checkpoints, search repos, manage authentication. Use when the user wants to download a model, upload to the Hub, or run any hf CLI command.
tools: Bash, Read, Edit
---

# HuggingFace Hub Skill

Handle HuggingFace Hub operations using the `hf` CLI and `huggingface_hub` Python API.

**Critical**: Always use `hf` — `huggingface-cli` is deprecated and no longer works.

## CLI Reference (`hf`)

```bash
hf auth login                          # authenticate
hf auth status                         # check login status

hf download <repo-id> [file]           # download full repo or single file
hf download <repo-id> <file> --local-dir <path>   # to specific directory
hf download <repo-id> --repo-type dataset         # dataset repo

hf upload <repo-id> <local-path> [dest-path]      # upload files
hf upload <repo-id> . .                            # upload entire directory

hf models ls --search "<query>"        # search models
hf datasets ls --search "<query>"      # search datasets
hf repos ls --format json              # list your repos
```

## Common Patterns for This Project

### Download a single SAE checkpoint file
```bash
hf download google/gemma-scope-2b-pt-res \
  layer_12/width_16k/average_l0_72/params.npz \
  --local-dir scratch/gemma-scope/
```

### Download a full model repo
```bash
hf download EleutherAI/gpt2-small-res-jb --local-dir scratch/gpt2-saes/
```

### Push a checkpoint to the Hub
```bash
hf upload <your-org>/<repo-name> scratch/checkpoint.safetensors checkpoints/checkpoint.safetensors
```

## Python API

When adding Hub operations to source files, use `huggingface_hub` functions directly:

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Single file
path = hf_hub_download(repo_id="google/gemma-scope-2b-pt-res",
                       filename="layer_12/width_16k/average_l0_72/params.npz",
                       local_dir="scratch/gemma-scope")

# Full repo snapshot
path = snapshot_download(repo_id="EleutherAI/gpt2-small-res-jb",
                         local_dir="scratch/gpt2-saes")
```

## Docstring / Comment Updates

If you see `huggingface-cli` in any docstring, comment, or README, replace it with the equivalent `hf` command. Example:

```
# Before (deprecated):
huggingface-cli download google/gemma-scope-2b-pt-res layer_12/.../params.npz --local-dir scratch/

# After:
hf download google/gemma-scope-2b-pt-res layer_12/.../params.npz --local-dir scratch/
```

## Workflow

1. **Check auth** — `hf auth status` before any download/upload; prompt user to run `! hf auth login` if not logged in (interactive, user must run it themselves)
2. **Download** — prefer `--local-dir` so files land in a predictable location (e.g., `scratch/`)
3. **Upload** — confirm the target repo and path before pushing
4. **Update stale references** — grep the codebase for `huggingface-cli` and update to `hf`
