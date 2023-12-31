#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8192M   # memory per CPU core
#SBATCH --qos=cs


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

python3 ../train_model.py \
    --activation-dropout 0.3 \
    --checkpoint-activations False \
    --embed-dim 128 \
    --ffn-dim 512 \
    --fsdp False \
    --layers 4 \
    --lr 0.0001 \
    --model enc_dec \
    --heads 8 \
    --seq-len 256 \
    --value-embed-dim 16 \
    --vocab-size 20000 \
    --device cuda \
    --batch-size 256 \
    --epochs 5 \
