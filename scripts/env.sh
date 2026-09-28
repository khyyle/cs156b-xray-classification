# Shared shell environment for job scripts and interactive helpers.
# Source this from anywhere inside the repo:
#
#   source "scripts/env.sh"
#
# Loads .env, activates the venv, and maps the CLUSTER_* settings onto
# Slurm's own input variables so plain `sbatch`/`srun` pick them up.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

# require all variables to be set in root .env
for required_setting in $(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$REPO_ROOT/.env.example"); do
    if [[ -z "${!required_setting:-}" ]]; then
        echo "env.sh: $required_setting is not set, set it in a root .env following the schema in .env.example" >&2
        return 1
    fi
done

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

export SLURM_ACCOUNT="$CLUSTER_ACCOUNT" SBATCH_ACCOUNT="$CLUSTER_ACCOUNT"
export SBATCH_PARTITION="$CLUSTER_GPU_PARTITION" # cpu jobs override this by passing partition explicitly

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Compute nodes have no outbound internet, so models must already be in the local HF cache to work
    export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
    export HF_HUB_OFFLINE=1
    # Ensembles with many members and TTA passes exceed the default 1024
    # open-file limit during DataLoader worker startup.
    ulimit -n 65536
fi
