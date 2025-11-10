from contextlib import suppress
import sys

import einops
from interbotix_common_modules.common_robot.exceptions import InterbotixException
from interbotix_common_modules.common_robot.robot import create_interbotix_global_node
from interbotix_common_modules.common_robot.robot import get_interbotix_global_node
from interbotix_common_modules.common_robot.robot import robot_startup
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import pathlib
import json

sys.path.append("/home/aloha/workspace/openpi")

from examples.aloha_mobile_real import real_env as _real_env

# from . import real_env as _real_env

def get_task_to_index_mapping(file_path: pathlib.Path) -> dict:
    mapping = {}
    with file_path.open('r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_description = record.get("task")
            task_index = record.get("task_index")
            if task_description is not None and task_index is not None:
                mapping[task_description] = task_index
    return mapping

class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        render_height: int = 224,
        render_width: int = 224,
        prompt: str = "uncap the pen",
        task_json: str = "metadata/aloha_pen_uncap/tasks.jsonl",
    ) -> None:
        try:
            node = get_interbotix_global_node()
        except Exception:
            node = create_interbotix_global_node("aloha")
        self._env = _real_env.make_real_env(node=node, setup_robots=True, setup_base=True)

        with suppress(InterbotixException):
            robot_startup(node)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None

        # add task information  
        filename = pathlib.Path(task_json)
        task_description2index = get_task_to_index_mapping(filename)

        self.prompt = prompt
        self.task_index = task_description2index[prompt]
        # import ipdb; ipdb.set_trace()

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        for cam_name in obs["images"]:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "images": obs["images"],
            "prompt": self.prompt,
            "task_index": self.task_index,
            "split": "test",
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action=action["actions"])
