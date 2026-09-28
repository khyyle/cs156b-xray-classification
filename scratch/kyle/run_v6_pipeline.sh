#!/bin/bash
# Single SLURM job that runs the full v6 ensemble pipeline once the
# ensemble training job (passed via --dependency on submission) finishes.
#
# Steps inside one job:
#   1, 2: cached test inference for public + private (uses the GPU because
#         submit.py auto-detects SLURM_JOB_ID and runs inline)
#   3-7: public per-disease NNLS, Ridge, per-view convex on each, blend
#   8-12: same for private
#
# All v5 NNLS .npy weights are written into V6 (the run dir is owned by the
# submitting user so writes succeed there directly); the public/private
# suffix is only applied if the run dir falls back to user_outputs.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=v6-pipeline
#SBATCH --output=runs/%u_outputs/v6_pipeline.out

set -e
source scripts/env.sh

V6=$(ls -d runs/ensemble-kitchen-sink-v6_*/ | head -1 | sed 's:/$::')
OUT="runs/${USER}_outputs"
echo "v6 run dir: $V6"

echo "=== 1. Public test inference ==="
python -u scripts/submit.py "$V6" --view-average

echo "=== 2. Private test inference ==="
python -u scripts/submit.py "$V6" --view-average --private-set

echo "=== 3. Public NNLS ==="
python -u scripts/per_disease_weights.py "$V6" --method nnls --submit

echo "=== 4. Public Ridge ==="
python -u scripts/per_disease_weights.py "$V6" --method ridge --ridge-alpha 1.0 --submit

echo "=== 5. Public per-view convex on NNLS-combined ==="
python -u scripts/per_view_fusion.py "$V6" --submit --convex \
  --per-disease-weights "$V6/per_disease_weights_nnls.npy"

echo "=== 6. Public per-view convex on Ridge-combined ==="
python -u scripts/per_view_fusion.py "$V6" --submit --convex \
  --per-disease-weights "$V6/per_disease_weights_ridge.npy"

echo "=== 7. Public blend ==="
V6_NAME=$(basename "$V6")
NNLS_PUB="$V6/${V6_NAME}_submission_per_view_convex_x_per_disease_weights_nnls_PUBLIC.csv"
RIDGE_PUB="$V6/${V6_NAME}_submission_per_view_convex_x_per_disease_weights_ridge_PUBLIC.csv"
python -u scripts/blend_submissions.py "$NNLS_PUB" "$RIDGE_PUB" \
  --output "$OUT/ensemble-kitchen-sink-v6_nnls_ridge_5050_per_view_convex_PUBLIC.csv"

echo "=== 8. Private NNLS ==="
python -u scripts/per_disease_weights.py "$V6" --method nnls --submit --private-set

echo "=== 9. Private Ridge ==="
python -u scripts/per_disease_weights.py "$V6" --method ridge --ridge-alpha 1.0 --submit --private-set

echo "=== 10. Private per-view convex on NNLS-combined ==="
python -u scripts/per_view_fusion.py "$V6" --submit --private-set --convex \
  --per-disease-weights "$V6/per_disease_weights_nnls.npy"

echo "=== 11. Private per-view convex on Ridge-combined ==="
python -u scripts/per_view_fusion.py "$V6" --submit --private-set --convex \
  --per-disease-weights "$V6/per_disease_weights_ridge.npy"

echo "=== 12. Private blend ==="
NNLS_PRIV="$V6/${V6_NAME}_submission_per_view_convex_x_per_disease_weights_nnls_PRIVATE.csv"
RIDGE_PRIV="$V6/${V6_NAME}_submission_per_view_convex_x_per_disease_weights_ridge_PRIVATE.csv"
python -u scripts/blend_submissions.py "$NNLS_PRIV" "$RIDGE_PRIV" \
  --output "$OUT/ensemble-kitchen-sink-v6_nnls_ridge_5050_per_view_convex_PRIVATE.csv"

echo "=== DONE ==="
