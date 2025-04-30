# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## COPY META DATA
copy the metadata from ```dingj0b@glogin.ibex.kaust.edu.sa:/home/dingj0b/code/openpi/metadata```

cp -r /home/dingj0b/code/openpi/metadata /home/dingj0b/dingjian/openpi_explore_storage//project/openpi/metadata

## Change ckp. Save Directory
Under config.py TrainConfig: checkpoint_base_dir:

"/ibex/tmp/c2090/openpi_explore_storage//checkpoints"

## TRAINING
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_incontext.py pi0_fast_libero --exp-name=my_experiment --overwrite
```
Replace "pi0_fast_libero" with config from config.py (e.g., pi0_libero_incontext_low_mem_finetune_sample2)
## TESTING

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main_incontext.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy_incontext.py --env LIBERO
```

Replace LIBERO with corresponding ENV from serve_policy_incontext.py (e.g., LIBERO_FM_LORA_INCONTEXT_SAMPLE2)

## Results

If you follow the training instructions and hyperparameters in the `pi0_libero` and `pi0_fast_libero` configs, you should get results similar to the following:

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0-FAST @ 30k (finetuned) | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π0 @ 30k (finetuned) | 96.8 | 98.8 | 95.8 | 85.2 | 94.15 |

Note that the hyperparameters for these runs are not tuned and $\pi_0$-FAST does not use a FAST tokenizer optimized for Libero. Likely, the results could be improved with more tuning, we mainly use these results as an example of how to use openpi to fine-tune $\pi_0$ models on a new dataset.
