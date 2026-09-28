#!/bin/bash
# Interactive GPU session for debugging models, running notebooks, etc.
#
# Usage:
#   bash scripts/srun_gpu.sh
#   bash scripts/srun_gpu.sh --gres=gpu:h100:1 --time=04:00:00
#   bash scripts/srun_gpu.sh --gres=gpu:v100:2 --mem=64G

set -euo pipefail
source "$(dirname "$0")/env.sh"

GRES=gpu:v100:1
CPUS=4
MEM=32G
TIME=02:00:00

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gres=*)
            GRES="${1#*=}"
            shift
            ;;
        --gres)
            GRES="$2"
            shift 2
            ;;
        --cpus-per-task=*)
            CPUS="${1#*=}"
            shift
            ;;
        --cpus-per-task)
            CPUS="$2"
            shift 2
            ;;
        --mem=*)
            MEM="${1#*=}"
            shift
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --time=*)
            TIME="${1#*=}"
            shift
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

srun --account="$CLUSTER_ACCOUNT" \
     --partition="$CLUSTER_GPU_PARTITION" \
     --nodes=1 \
     --ntasks=1 \
     --gres="$GRES" \
     --cpus-per-task="$CPUS" \
     --mem="$MEM" \
     --time="$TIME" \
     --pty bash
