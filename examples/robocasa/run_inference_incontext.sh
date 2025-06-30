#!/bin/bash

# 3690705
RUN_NAME=${1:-XJ_PI0_MINI_ROBOCASA_MG_INCONTEXT}
IP=${2:-"127.0.0.1"}
OPENPI_SCRIPT=${3:-serve_policy_incontext.py}
PYTHON_SCRIPT=${4:-main_incontext.py}
# XJ_PI0_MINI_ROBOCASA_MG_INCONTEXT XJ_PI0MINI_ROBOCASA_MG_THREE_IMAGE

LOG_DIR="/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/logs/${RUN_NAME}"

mkdir -p "$LOG_DIR"

# # pi0 server code
cd /home/dingj0b/dingjian/openpi_explore/project/openpi

uv run scripts/${OPENPI_SCRIPT} --env "$RUN_NAME" > ${LOG_DIR}/server.log 2>&1 &

# inference code
cd /home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa

export PYTHONPATH=$PYTHONPATH:/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robomimic
export PYTHONPATH=$PYTHONPATH:/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robosuite


# module load anaconda3
eval "$(conda shell.bash hook)"
conda activate robocasa

echo "Current conda environment: $CONDA_DEFAULT_ENV"
which python

# pip install tyro


# 目标 Python 脚本


# 定义 human_im_dir 中的全部路径（每一行是一个 demo）
hdf5_paths=(
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-05-04-22-21-32/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-05-04-22-40-00/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-05-04-22-39-23/demo_gentex_im128_randcams.hdf5'
         
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-07-12-04-33-29/demo_gentex_im128_randcams.hdf5'
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_gentex_im128_randcams.hdf5'
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-05-04-22-13-21_and_2024-05-07-07-41-17/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-05-04-22-14-26_and_2024-05-07-07-41-42/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-05-04-22-14-20/demo_gentex_im128_randcams.hdf5'
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/mg/2024-05-04-22-14-40/demo_gentex_im128_randcams.hdf5' 
         
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/mg/2024-05-04-22-37-39/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/2024-05-04-22-34-56/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/mg/2024-05-04-22-35-53/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/mg/2024-05-04-22-22-42_and_2024-05-08-06-02-36/demo_gentex_im128_randcams.hdf5' 
         
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/mg/2024-05-04-22-38-42/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/mg/2024-05-09-09-32-19/demo_gentex_im128_randcams.hdf5' 
         
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-05-04-22-17-46/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-05-04-22-17-26/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/mg/2024-05-09-09-31-12/demo_gentex_im128_randcams.hdf5' 
         
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/mg/2024-05-08-09-20-31/demo_gentex_im128_randcams.hdf5' 
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/mg/2024-05-08-09-20-45/demo_gentex_im128_randcams.hdf5'
         
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-05-04-22-22-13_and_2024-05-08-05-52-13/demo_gentex_im128_randcams.hdf5'
         '/home/dingj0b/dingjian/openpi_explore/project/robocasa_family/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-05-04-22-21-50/demo_gentex_im128_randcams.hdf5'
)

which python

# 循环执行
for path in "${hdf5_paths[@]}"; do
    env_name=$(basename "$(dirname "$(dirname "$(dirname "$path")")")")
    echo "Running $env_name with $path"
    python "$PYTHON_SCRIPT" --args.dataset-path "$path" --args.env-name "$env_name" > ${LOG_DIR}/${env_name}.log 2>&1
done

echo "All tasks finished. Killing ${OPENPI_SCRIPT} to free resources..."
pkill -f ${OPENPI_SCRIPT}