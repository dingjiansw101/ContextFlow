#!/bin/bash

RUN_NAME=XJ_ROBOCASA_INFERENCE

LOG_DIR="/home/xianjie.dai/project/debug/robocasa/logs/${RUN_NAME}"

mkdir -p "$LOG_DIR"

# pi0 server code
cd /home/xianjie.dai/project/openpi

uv run scripts/serve_policy_incontext.py --env "$RUN_NAME" > ${LOG_DIR}/server.log 2>&1 &

# inference code
cd /home/xianjie.dai/project/debug/robocasa

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/jian/xianjie/conda_envs/robocasa

# pip install tyro

export PYTHONPATH=$PYTHONPATH:/home/xianjie.dai/project/debug/robosuite
export PYTHONPATH=$PYTHONPATH:/home/xianjie.dai/project/debug/robocasa



# python /mnt/jian/xianjie/project/robomimic/robomimic/scripts/setup_macros.py

# 目标 Python 脚本
PYTHON_SCRIPT="main.py"

# 定义 human_im_dir 中的全部路径（每一行是一个 demo）
hdf5_paths=(
    # "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29/demo.hdf5"

    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03/demo.hdf5"
    # "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo.hdf5"

    # "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29/demo.hdf5"

    # "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02/demo.hdf5"
    
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo.hdf5"

    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo.hdf5"
    # "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25/demo.hdf5"

    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_navigate/NavigateKitchen/2024-05-09/demo.hdf5"

    # "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo.hdf5"
    # "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27/demo.hdf5"
    # "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26/demo.hdf5"
    "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo.hdf5"
    # "/home/xianjie.dai/project/debug/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo.hdf5"
)

# 循环执行
for path in "${hdf5_paths[@]}"; do
    env_name=$(basename "$(dirname "$(dirname "$path")")")
    echo "Running $env_name with $path"
    python "$PYTHON_SCRIPT" --args.dataset-path "$path" --args.env-name "$env_name" > ${LOG_DIR}/${env_name}.log 2>&1
done

