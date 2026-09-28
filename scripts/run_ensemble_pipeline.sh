#!/bin/bash
# Full submission pipeline for one ensemble run dir, public + private.
# Generalizes run_v6_pipeline.sh to any ensemble: pass the run dir as $1.
#
# Steps (all in one GPU job; submit.py runs inline because SLURM_JOB_ID is set):
#   public:  test inference -> per-disease NNLS + Ridge -> per-view convex on
#            each -> 50/50 blend
#   private: same
#
# Final blended CSVs land in runs/<user>_outputs/<run_name>_..._{PUBLIC,PRIVATE}.csv
#
# Usage:
#   mkdir -p "runs/${USER}_outputs"
#   source scripts/env.sh
#   sbatch --export=ALL,RUN_DIR=runs/ensemble-kitchen-sink-v7-full_<ts> \
#       scripts/run_ensemble_pipeline.sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=ens-pipeline
#SBATCH --output=runs/%u_outputs/ensemble_pipeline_%j.out

set -e
source scripts/env.sh

if [ -z "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR env var not set. Pass via --export=ALL,RUN_DIR=..."
  exit 1
fi
V=$(echo "$RUN_DIR" | sed 's:/$::')
NAME=$(basename "$V")
OUT="runs/${USER}_outputs"
echo "Ensemble run dir: $V"

echo "=== Public test inference ==="
python -u scripts/submit.py "$V" --view-average

echo "=== Private test inference ==="
python -u scripts/submit.py "$V" --view-average --private-set

for phase in public private; do
  if [ "$phase" = "private" ]; then private_flag="--private-set"; suffix="PRIVATE"; else private_flag=""; suffix="PUBLIC"; fi
  echo "=== $phase: per-disease NNLS ==="
  python -u scripts/per_disease_weights.py "$V" --method nnls --submit $private_flag
  echo "=== $phase: per-disease Ridge ==="
  python -u scripts/per_disease_weights.py "$V" --method ridge --ridge-alpha 1.0 --submit $private_flag
  echo "=== $phase: per-view convex (NNLS) ==="
  python -u scripts/per_view_fusion.py "$V" --submit $private_flag --convex \
    --per-disease-weights "$V/per_disease_weights_nnls.npy"
  echo "=== $phase: per-view convex (Ridge) ==="
  python -u scripts/per_view_fusion.py "$V" --submit $private_flag --convex \
    --per-disease-weights "$V/per_disease_weights_ridge.npy"
  echo "=== $phase: blend ==="
  NNLS_CSV="$V/${NAME}_submission_per_view_convex_x_per_disease_weights_nnls_${suffix}.csv"
  RIDGE_CSV="$V/${NAME}_submission_per_view_convex_x_per_disease_weights_ridge_${suffix}.csv"
  python -u scripts/blend_submissions.py "$NNLS_CSV" "$RIDGE_CSV" \
    --output "$OUT/${NAME}_nnls_ridge_5050_per_view_convex_${suffix}.csv"
done

echo "=== DONE ==="
