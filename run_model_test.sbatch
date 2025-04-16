#!/bin/bash
#SBATCH --job-name=flux_generate
#SBATCH --output=logs/flux_generate_%j.out
#SBATCH --error=logs/flux_generate_%j.err
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
  --bind /home/s497288/lab-project/models:/models \
  ~/lab-project/model_test_sandbox \
  --output-dir /images \
  --model-path /models/FLUX.1-dev \
  --plan generation_plan.json \
  --steps 16 \
  --batch-size 16



#FOR SLURM

1.) sed -i 's/\r$//' run_model_test.sbatch
2.) sbatch run_model_test.sbatch


O1.) cat logs/flux_generate_394663.out
O2.) squeue -j 394663