#!/bin/bash

#SBATCH --account=def-wperciva
#SBATCH --job-name=test1
#SBATCH --output=out_test1.out
#SBATCH --time=500:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000M

module load julia/1.10.0

srun julia --project=/home/acrespi/FILEBAIN -t 4 /home/acrespi/FILEBAIN/test1.jl
