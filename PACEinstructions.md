# Running LLM Guided Evolution (Titanic) on PACE ICE

This guide walks you through running the `MosesTheRedSea-main` branch of the LLM Guided Evolution project on Georgia Tech's PACE ICE cluster.

---

## Prerequisites

- Access to PACE ICE (`login-ice.pace.gatech.edu`)
- The repo already cloned on ICE
- A [Kaggle account](https://www.kaggle.com) with an API token (for downloading the dataset)

---

## Step 1 — SSH in and navigate to the repo

```bash
ssh <your-gt-username>@login-ice.pace.gatech.edu
cd <path-to-repo>/llm-guided-evolution-fork
```

Make sure you're on the right branch:

```bash
git branch
# Should show: * MosesTheRedSea-main
```

If not, switch to it:

```bash
git checkout MosesTheRedSea-main
```

---

## Step 2 — Set up the environment (first time only)

```bash
module load uv
uv sync
```

This installs all dependencies into a local `.venv` using the locked versions in `uv.lock`.

---

## Step 3 — Download the Titanic dataset (first time only)

First, set up your Kaggle API credentials if you haven't already:

1. Go to [kaggle.com](https://www.kaggle.com) → Account → **Create New API Token**
2. This downloads a `kaggle.json` file. Copy its contents to ICE:

```bash
mkdir -p ~/.kaggle
# Paste your token details:
echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Then download the data:

```bash
cd sota/Titanic
uv run kaggle competitions download -c titanic -p data -f train.csv
cd ../..
```

> If `data/train.csv` already exists from a previous run, skip this step.

---

## Step 4 — Generate the SLURM scripts

This reads the PACE ICE cluster config and writes the correct SBATCH headers into `run.sh`, `server.sh`, and `src/mixt.sh`:

```bash
uv run python slurm.py
```

Quickly verify the output looks right:

```bash
head -12 server.sh
head -12 run.sh
```

---

## Step 5 — Submit the LLM server job first

The evolution loop makes LLM calls to a local server, so the server **must be running before the evolution starts**.

```bash
sbatch server.sh
```

Watch for it to reach **R** (Running) status:

```bash
watch squeue -u $USER
```

Once it's running, confirm it wrote its hostname:

```bash
cat hostname.log
# Should print something like: atl1-1-03-012-23-0
```

---

## Step 6 — Submit the evolution job

```bash
sbatch run.sh
```

This runs `run_improved.py titanic_test` and automatically submits sub-jobs (LLM mutations, crossovers, evaluations) via `sbatch` as the evolution proceeds. You'll see many `llm_oper` jobs appear in the queue — that's normal.

---

## Step 7 — Monitor progress

```bash
# View all your active jobs
watch squeue -u $USER

# Tail the main evolution log (replace JOBID with your run.sh job ID)
tail -f slurm-<JOBID>.out

# Count how many model variants have been generated
watch -n 5 'ls sota/Titanic/models/llmge_models/ | wc -l'
```

---

## Troubleshooting

**`kaggle: command not found`**
The kaggle CLI lives inside the uv environment. Use `uv run kaggle ...` instead of `kaggle ...`.

**`llm_oper` jobs stuck on `BadConstraints`**
The GPU constraint in the generated script doesn't match available node features. Check what's available with:
```bash
sinfo -p ice-gpu -o "%N %f" | head -30
```
Then update the `llm-gpu` section in `slurm-config/pace-ice.txt` to match real feature names and re-run `uv run python slurm.py`.

**Server job running but evolution can't connect**
Make sure `hostname.log` exists and contains a valid hostname before submitting `run.sh`. The evolution reads this file to find the server.

**Jobs disappearing before finishing**
The default time limit is 8 hours. If your run needs longer, edit the `#SBATCH -t` line in `slurm-config/pace-ice.txt` and re-run `slurm.py`.

---

## Key configuration (`src/cfg/constants.py`)

| Setting | Value |
|---|---|
| Cluster | `pace-ice` |
| LLM | Llama 3.3 70B (local, at `/storage/ice-shared/vip-vvk/`) |
| Server port | `8137` |
| Output directory | `titanic_test` |
| Time limit | 8 hours per job |
| Submission mode | `sbatch` (fully automated sub-job submission) |

---

## Quick reference

```bash
# One-time setup
module load uv && uv sync
cd sota/Titanic && uv run kaggle competitions download -c titanic -p data -f train.csv && cd ../..

# Every run
uv run python slurm.py
sbatch server.sh        # wait for it to show R in squeue
sbatch run.sh
```
