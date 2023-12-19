#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8192M   # memory per CPU core
#SBATCH --qos=cs


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

python3 ../train_model.py \
    --activation-dropout 0.1 \
    --checkpoint-activations False \
    --dropout 0.0 \
    --embed-dim 16 \
    --ffn-dim 16 \
    --fsdp True \
    --layers 2 \
    --lr 0.001 \
    --model enc_dec \
    --heads 4 \
    --seq-len 256 \
    --value-embed-dim 16 \
    --vocab-size 1000 \
    --device cuda \
    --batch-size 1 \
    --epochs 10000 \
