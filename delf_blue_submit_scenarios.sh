#!/bin/sh

#SBATCH --job-name="IGAD_scenarios"
#SBATCH --partition=compute
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=research-tpm-mas

module load 2022r2

srun python delf_blue_run_scenarios.py
