#!/bin/bash
# SLURM job: SmolVLA on the LIBERO-plus *language-perturbation* tasks.
#
# Submit from giano.cs.unibo.it, from the folder that contains this file:
#     sbatch run_eval.sbatch                                          # all language tasks of libero_spatial
#     sbatch --export=ALL,MAX_TASKS=2 run_eval.sbatch                 # quick 2-task test
#     sbatch --export=ALL,SUITE=libero_goal run_eval.sbatch           # another suite
#
# Before submitting: replace `nome.cognome` in --mail-user and --chdir below (SLURM does not expand
# variables in #SBATCH lines) and make sure `logs/` exists inside that --chdir folder (setup_cluster.sh does).
# NB: type the directive dashes as plain ASCII "--"; the cluster PDF shows some as typographic dashes.
#
#SBATCH --job-name=smolvla-lang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.preda2@studio.unibo.it
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=rtx2080
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch.hpc/matteo.preda/tesi
#SBATCH --gres=gpu:1

set -eo pipefail

# ---- paths (must match setup_cluster.sh) ----
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch.hpc/matteo.preda/tesi}"   # keep in sync with setup_cluster.sh
export PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"     # folder you ran `sbatch` from: it holds smolvla_text_perturbation.py + the launcher
export PATH="$SCRATCH_ROOT/venv/bin:$PATH"

# ---- keep every cache out of the 400 MB home quota ----
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export XDG_CACHE_HOME="$SCRATCH_ROOT/.cache"
export TMPDIR="$SCRATCH_ROOT/tmp"
export MPLCONFIGDIR="$SCRATCH_ROOT/.mplconfig"
mkdir -p "$TMPDIR"

# ---- runtime ----
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"        # weights were cached by setup_cluster.sh; set to 0 if the nodes have internet
export TRANSFORMERS_OFFLINE="$HF_HUB_OFFLINE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ---- experiment parameters (override with: sbatch --export=ALL,NAME=value ...) ----
export SUITE="${SUITE:-libero_spatial}"                  # libero_spatial | libero_object | libero_goal | libero_10
export POLICY_PATH="${POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
export MAX_TASKS="${MAX_TASKS:-0}"                       # 0 = all language tasks of the suite
export CHUNK_SIZE="${CHUNK_SIZE:-25}"                    # tasks per lerobot-eval call; finished chunks are skipped on re-run
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
export SEED="${SEED:-1000}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-}"              # empty = checkpoint default (1 for HuggingFaceVLA/smolvla_libero)
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/results}"
SUFFIX=""; [ "$MAX_TASKS" != "0" ] && SUFFIX="_first${MAX_TASKS}"   # a test run must not share a folder with the full run
export RUN_NAME="${RUN_NAME:-smolvla_lang_${SUITE}${SUFFIX}}"

echo "job $SLURM_JOB_ID on $(hostname) | suite=$SUITE max_tasks=$MAX_TASKS chunk=$CHUNK_SIZE | $(python --version)"
nvidia-smi -L

cd "$PROJECT_DIR"
python -u smolvla_text_perturbation.py      # -u: unbuffered, so the SLURM log streams live
echo "finished: $OUTPUT_DIR/$RUN_NAME/results.json"
