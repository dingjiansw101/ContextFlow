#!/bin/bash
#SBATCH --job-name=pi0tiny_incontext_multi_2x8
#SBATCH --output=logs/pi0tiny_incontext_multi_2x8_%x-%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --nodes=2                 
#SBATCH --ntasks=2                
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8         
#SBATCH --cpus-per-task=80       
#SBATCH --exclusive               

export WANDB_API_KEY=4da092c90a4ccc6ce17c17c6c5d268bbd9b628f2
export COORDINATOR_PORT=12355     

cd ..

srun bash -lc '
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext_multi_node.py \
        pi0_libero_low_mem_finetune_split_train --project-name=pi0_libero_low_mem_finetune_split_train_dummy \
        --exp-name=pi0_libero_low_mem_finetune_split_train_dummy \
        --save_interval=10_000 \
        —overwrite
'