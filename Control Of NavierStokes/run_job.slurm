#!/bin/bash
#SBATCH --job-name=python-job
#SBATCH --ntasks=40
#SBATCH --time=0-1:00:00
#SBATCH --partition=intel
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

export PYTHONUNBUFFERED=1
export MPI_TASK_GROUP_SIZE=1
export MPI_PARALLEL_FD=1

SCRIPT=$1

srun --mpi=pmi2 \
/stck/eclement/.conda/envs/petsc-cplx/bin/python -u "$SCRIPT"
