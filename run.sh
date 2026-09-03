#!/bin/bash
#SBATCH --job-name=sma_branch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =========================
# Change only here
# =========================
SUBJECT_ID=""
STUDY_ID=""
SERIES_ID=""

# =========================
# Automatically generated filename/path
# =========================
CASE_ID="${SUBJECT_ID}_${STUDY_ID}_${SERIES_ID}"

CT= "raw CT image"
SEG="SMA & SMV mask"
OUT="${CASE_ID}_gt.nii.gz"
BOWEL_SEG="Small Bowel mask"

python -u gen_vessels.py \
  --ct "$CT" \
  --seg "$SEG" \
  --out "$OUT" \
  --hu_min 50 \
  --hu_max 370 \
  --vesselness_thr 0.001 \
  --margin 95 \
  --max_iter 700 \
  --min_size 25 \
  --air_hu -500 \
  --air_radius 2 \
  --bowel_seg "$BOWEL_SEG" \
  --bowel_interior_radius 2