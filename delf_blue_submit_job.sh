#!/bin/sh

#SBATCH --job-name="IGAD_model"
#SBATCH --partition=compute
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --account=research-tpm-mas

module load 2022r2
module load miniconda3
conda activate igad_test_py39

echo $1 $2 $3 $4 $5 $6
srun python batch_run.py $1 $2 $3 $4 $5 $6

conda deactivate
