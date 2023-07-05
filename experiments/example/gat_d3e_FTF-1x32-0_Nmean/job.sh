#!/bin/bash
#SBATCH --job-name="gat_d3e_FTF-1x32-0_Nmean"
#SBATCH --partition=v100-32g
###SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --nodelist=hild
###SBATCH --nodelist=ootaki
###SBATCH --nodelist=tamiya
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00
#SBATCH --chdir=./
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt

sbatch_pre.sh

module load python/3.8.10-gpu-cuda-11.1

# pip install virtualenv

# python3 -m venv ~/venv/thermo
source ~/venv/thermo/bin/activate

# pip install --upgrade pip
# pip install torch_geometric wandb torchinfo torch tqdm transformers matplotlib numpy prody numpy

# wandb agent --count 14 yenlin-chen/thermostability/68zu3il3
python3 train.py

sbatch_post.sh
