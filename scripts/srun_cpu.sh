#!/bin/bash
# Interactive CPU session for lightweight tasks.
#
# Usage:
#   bash scripts/srun_cpu.sh
#   bash scripts/srun_cpu.sh --cpus-per-task=8 --mem=32G --time=02:00:00

set -euo pipefail
source "$(dirname "$0")/env.sh"

CPUS=4
MEM=16G
TIME=01:00:00

while [[ $# -gt 0 ]]; do
    case "$1" in
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
     --partition="$CLUSTER_CPU_PARTITION" \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task="$CPUS" \
     --mem="$MEM" \
     --time="$TIME" \
     --pty bash
