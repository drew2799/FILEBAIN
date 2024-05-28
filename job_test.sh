#!/bin/bash

#SBATCH --job-name=test1
#SBATCH --output=out_test1.out
#SBATCH --time=15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1024M

module load julia/1.10.0

srun julia --project=/home/acrespi/FILEBAIN /home/acrespi/FILEBAIN/test1.jl
