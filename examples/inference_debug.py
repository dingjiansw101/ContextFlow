import dataclasses

from aloha_mobile_real.convert_aloha_mobile_data_to_lerobot import load_raw_episode_data
import einops
from openpi_client import image_tools
import torch

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

config = _config.get_config("pi0_fast_aloha_pen_uncap_b5")
checkpoint_dir = "/home/dingj0b/code/openpi/checkpoints/pi0_fast_aloha_pen_uncap_b5/pi0_fast_aloha_pen_uncap_b5/9999"
# mse loss: 3.2497e-05
# mse loss with error decoding tokens: 0.0276

# config = _config.get_config("pi0_aloha_pen_uncap_b5_low_mem_finetune")
# checkpoint_dir = "/home/dingj0b/code/openpi/checkpoints/pi0_aloha_pen_uncap_b5_low_mem_finetune/pi0_aloha_pen_uncap_b5_low_mem_finetune/10000"
# mse loss: 0.0002

policy = _policy_config.create_trained_policy(config, checkpoint_dir)


# Reduce the batch size to reduce memory usage.
config = dataclasses.replace(config, batch_size=4)

# Load a single batch of data. This is the same data that will be used during training.
# NOTE: In order to make this example self-contained, we are skipping the normalization step
# since it requires the normalization statistics to be generated using `compute_norm_stats`.
# import ipdb; ipdb.set_trace()
# loader = _data_loader.create_data_loader(config, num_batches=4, skip_norm_stats=False)
# obs, act = next(iter(loader))

imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(
    "/home/dingj0b/datasets/trossen_aloha_real/pen_uncap_b5/episode_0.hdf5"
)


def image_transform(img):
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    return einops.rearrange(img, "h w c -> c h w")


# data_config = config.data.create(config.assets_dirs, config.model)
# dataset = create_dataset(data_config, config.model)
# Sample actions from the model.
# option 1: use dataset without transform, check the lerobot dataset
# option 2: directly use model.action_samples, then use output transform
# for i in range(len(dataset)):
#     data_config.repack_transforms.inputs[0](dataset[i])
# data_config.repack_transforms.inputs[0](obs.to_dict())
for i in range(imgs_per_cam["cam_high"].shape[0]):
    obs = {
        "state": state[i],
        "images": {
            # check, if permute is enough for the transformation on images
            "cam_high": image_transform(imgs_per_cam["cam_high"][i]),
            "cam_left_wrist": image_transform(imgs_per_cam["cam_left_wrist"][i]),
            "cam_right_wrist": image_transform(imgs_per_cam["cam_right_wrist"][i]),
        },
        "prompt": "uncap the pen",
    }

    # import ipdb; ipdb.set_trace()
    act_pred = policy.infer(obs)["actions"]  # predicted action for the next 32 steps
    act_pred = torch.from_numpy(act_pred)
    # import ipdb; ipdb.set_trace()
    print("i: ", i)
    act_gt = action[i : i + act_pred.shape[0]]  #

    mse = torch.mean((act_pred - act_gt) ** 2)
    print("mse: ", mse)
    # import ipdb; ipdb.set_trace()
# import ipdb; ipdb.set_trace()
# Delete the model to free up memory.
del policy

print("act_pred shape: ", act_pred.shape)
print("act_pred: ", act_pred)
# print("Loss shape:", loss.shape)
# print("Loss: ", loss)
