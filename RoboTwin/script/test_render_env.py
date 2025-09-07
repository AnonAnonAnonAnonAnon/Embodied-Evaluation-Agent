# add path of envs to the script
import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError


# set up the env name
import importlib

import yaml

def class_decorator(task_name):
    """
    from envs import a module named ${task_name}, which is an env
    and create an INSTANCE of this task env
    """
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

# define a function to get robot file
def get_embodiment_file(embodiment_type):
    """
    input:
        embodiment_type: robot type
    output:
        robot_file: robot file path 
    """

    # get available robot types and their paths
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    # return the robot file path that we want
    robot_file = _embodiment_types[embodiment_type]["file_path"]
    if robot_file is None:
        raise "No embodiment files"
    return robot_file

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def main():
    # orig instruction:
    # bash eval.sh beat_block_hammer demo_randomized 0 50 0 0

    kwargs = {
        "task_name": "beat_block_hammer",
        "task_config": "demo_randomized",
        "ckpt_setting": 0,
        "expert_data_num": 50,
        "seed": 0,
        "gpu_id": 0
    }

    # create a task instance
    TASK_ENV = class_decorator(kwargs['task_name']) 

    # load args from yaml
    config_path = f"./task_config/{kwargs['task_config']}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # get camera config
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    # and load camera config to args
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    # load args of embodiment(robot)
    embodiment_type = args.get("embodiment") # [aloha-agilex], which is a LIST
    # load robot config to args
    if len(embodiment_type) == 1:
        # the task support 1 kind of robot
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        # the task support 3 kinds of robot
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    
    now_id = 0
    now_seed = 0
    args['render_freq'] = 10 # force rendering
    print("args['render_freq']", args['render_freq'])
    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
    print(TASK_ENV.render_freq)
    print(TASK_ENV.viewer)

    # play one episode
    TASK_ENV.play_once()


if __name__ == '__main__':
    from test_render import Sapien_TEST
    Sapien_TEST()

    main()