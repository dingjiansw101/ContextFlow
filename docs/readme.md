# Config ssh keys for github & huggingface dataset repos

## Generate & Add SSH Key to Github & HuggingFace Accounts

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

```bash
ssh-keygen -t ed25519 -C "[your_email@example.com](mailto:your_email@example.com)"
eval "$(ssh-agent -s)”
ssh-add ~/.ssh/id_ed25519
```

```bash
cat ~/.ssh/id_ed25519.pub

# Add the public key to your GitHub account and HuggingFace account (for downloading data)
```

## Clone Repo & Set up Environment

```bash
git clone [git@github.com](mailto:git@github.com):dingjiansw101/openpi.git
cd openpi/
# install uv since openpi uses uv to manage environment
wget -qO- https://astral.sh/uv/install.sh | sh
# run “uv --version” to check whether uv is available

# IMPORTANT: By default, uv is installed to ~/.local/bin and in Linux,
# ~/.local/bin may not be added to PATH automatically.
# If it's your case, you can add it to PATH
# temporarily: 
# export PATH="$HOME/.local/bin:$PATH"
# permanantly: 
# echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc; source ~/.bashrc

# sync environment
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

# Train

## Preparation

There are several necessary files that need to be created manually. 

Assuming you are already in the project directory, i.e. xxx/openpi:

```bash
# where sbatch logs will be saved
mkdir logs 
# where pre-computed normalization statisctis is stored
mkdir assets
# where checkpoints will be saved
mkdir checkpoints
# where pre-computed training metadata is saved
mkdir metadata

# 1. add norm_stats.json 
cd assets
mkdir -p assets/pi0tiny_incontext_robocasa_mg_three_image_scaleup_train_split/daixianjie/robocasa_mg_lerobot

mkdir -p assets/pi0tiny_incontext_robocasa_mg_three_image_scaleup_inference/daixianjie/robocasa_mg_lerobot
# copy the following json file to the folders

2. metadata
# please copy the shared metadata (robocasa) to the "metadata" folder under the project directory

```

## Training Script (please avoid any V100 GPU!)

Here is an example using SLURM to submit a training job (assuming the script is under the project directory, i.e. xxx/openpi/scripts.sh):

```bash
#!/bin/bash
#SBATCH --**mem=200G** # memory pool for all cores`
#SBATCH --**time 24:00:00** # time, specify max time allocation`
#SBATCH --**gres=gpu:a100:8**
#SBATCH --**cpus-per-gpu=10**
#SBATCH --job-name=pi0tiny_incontext_robocasa_mg_three_image_scaleup_train_split
#SBATCH --output=logs/pi0tiny_incontext_robocasa_mg_three_image_scaleup_train_split_%x-%j.log

export WANDB_API_KEY=4da092c90a4ccc6ce17c17c6c5d268bbd9b628f2 

cd ../..

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_mini_incontext.py pi0tiny_incontext_robocasa_mg_three_image_scaleup_train_split --project-name=pi0tiny_incontext_robocasa_mg_three_image_scaleup_train_split --exp-name=pi0tiny_incontext_robocasa_mg_three_image_scaleup_train_split --save_interval=100_000 --**resume** 

```

Notes:

1. Training may last several days and this script only asks for 24 hours of training. 
    
    The command has been set to automatically resume from last saved checkpoint. 
    
    Only need to submit the script multiple times without any modification 
    
    Notice that resuming training requires the same amount of GPUs as the first training session, which means you have to modify “--**resume**” (at the end of last command) to “**—overwrite**” in order to start training from scratch (remember to modify it to “resume” when resume training).
    
2. Please modify the SBATCH parameters accordingly. The number of GPUs is recommended to be larger
3. The WANDB_API_KEY belongs to Xianjie for monitoring the training process.
4. Notice that robocasa dataset is enormous in size, around 1TB and the script will automatically download the dataset to the root directory:
    
    ```bash
    ~/.cache/huggingface/lerobot/daixianjie/robocasa_mg
    ```
    
    You may want to create a softlink that mount “~/.cache/huggingface/lerobot” to another location with larger storage space.
    
5. The script will generate 15 checkpoints under “xxx/openpi/checkpoints” directory, each of which is around 2-3 GB.