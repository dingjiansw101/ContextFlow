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
```

## Training Script

Please see the main CLAUDE.md documentation for training instructions.