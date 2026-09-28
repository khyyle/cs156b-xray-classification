#!/bin/bash
# Dependency job that runs the correlation diagnostic against the throwaway
# ensemble once it finishes caching val preds for all candidates.
# CPU-only, so submit with the CPU partition:
#   source scripts/env.sh
#   sbatch --partition="$CLUSTER_CPU_PARTITION" scratch/kyle/run_corr_check.sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=corr-check
#SBATCH --output=runs/%u_outputs/corr_check.out

set -e
source scripts/env.sh

ENS=$(ls -d runs/_corr_check_v4_plus_candidates_*/ | head -1 | sed 's:/$::')
echo "Diagnosing ensemble at $ENS"
echo

for focus in siglip2 omnirad asl-per-class; do
  echo "############################################################"
  echo "# focus-member: $focus"
  echo "############################################################"
  python -u scratch/kyle/investigate_ensemble_errors.py "$ENS" --focus-member "$focus"
  echo
done

echo "=== DONE ==="
