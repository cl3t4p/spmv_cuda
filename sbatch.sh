#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

#SBATCH --job-name=spmv_cuda
#SBATCH --output=output/spmv_cuda-%j.out
#SBATCH --error=output/spmv_cuda-%j.err

module load CUDA/11.8.0

./build/linux/x86_64/release/spmv "$@"
