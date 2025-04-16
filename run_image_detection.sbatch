#!/bin/bash
#SBATCH --job-name=image_detect
#SBATCH --output=logs/image_detect_%j.out
#SBATCH --error=logs/image_detect_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Optional: create logs dir if it doesn't exist
mkdir -p logs

module load apptainer

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer run --nv \
  --bind /home/s497288/lab-project/images:/images \
  --bind /home/s497288/lab-project/results:/results \
  ~/lab-project/image_detection_sandbox \
  --images-dir /images \
  --results-dir /results \
  --model-path /yolo11x.pt
