# add path of envs to the script
import sys
import os
import subprocess
import threading

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError


# set up the env name
import importlib

import yaml

# image operations
from PIL import Image

import subprocess

import cv2
import time

def save_all_frames(TASK_ENV, stop_event):
    """
    Save all frames captured during the task execution.
    """
    image_index = 0
    viewer = TASK_ENV.viewer

    lock = threading.Lock()
    while not stop_event.is_set() and not viewer.closed:
        # if viewer.window.key_down("p"):  # Press 'p' to take the screenshot
        print(f"screen shot {image_index}")
        try:
            rgba = viewer.window.get_picture("Color")
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            print("image shape: {}".format(rgba_img.shape) + f" with type {type(rgba_img)}")
            bgra_img_cv2 = rgba_img[:, :, [2, 1, 0, 3]]
            cv2.imwrite(f"script/temp_figs/screenshot_{image_index:03d}.png", bgra_img_cv2)
            image_index += 1
        except Exception as e:
            print(e)
        # rgb_cv2 = rgba_img[:, :, :3]
        # rgba_pil = Image.fromarray(rgba_img)
        # rgba_pil.save("screenshot.png")

        if viewer.window.key_down("q"):
            print('q pressed')
            TASK_ENV.close_env()
            break
        viewer.scene.step()
        viewer.scene.update_render()
        viewer.render()


def frames2mp4(task_name):
    import glob

    # read all images and sort
    img_files = sorted(glob.glob("script/temp_figs/screenshot_*.png"))

    # read the very first image to get h and w
    frame = cv2.imread(img_files[0])
    print("frame shape: {}".format(frame.shape) + f" with type {type(frame)}")
    h, w, _ = frame.shape

    # video writer
    out = cv2.VideoWriter(f'script/task_demos/{task_name}.avi',
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        5.0,  # fps
                        (w, h))

    # write all images to video
    for f in img_files:
        img = cv2.imread(f)
        out.write(img)

    out.release()

    # avi2mp4
    print(f"converting {task_name}.avi to {task_name}.mp4")
    subprocess.call(["ffmpeg", "-i", f"script/task_demos/{task_name}.avi", f"script/task_demos/{task_name}.mp4", '-y'])
    # remove the avi file
    subprocess.call(["rm", f"script/task_demos/{task_name}.avi"])

def render_task(TASK_ENV, task_name:str, setup_begin_time:float):
    """
    Execute the task and save frames in parallel.
    """
    # clear the temp_figs folder
    print("deleting all images in script/temp_figs")
    subprocess.call(["pwd"])
    # subprocess.call(["rm", "-rf", "./script/temp_figs/*"])
    subprocess.call(["rm", "-rf", "./script/temp_figs/"])
    subprocess.call(["mkdir", "-p", "./script/temp_figs/"])
    print("deletion completed")

    # Event to control the save_thread
    stop_event = threading.Event()

    # Create threads
    execution_begin_time = time.time()
    play_thread = threading.Thread(target=TASK_ENV.play_once)
    save_thread = threading.Thread(target=save_all_frames, args=(TASK_ENV, stop_event))

    # Start threads
    play_thread.start()
    save_thread.start()

    # Wait for play_thread to complete
    play_thread.join()
    stop_event.set()  # Signal save_thread to stop

    # Wait for save_thread to complete
    save_thread.join()

    # move the first image to task_screenshots
    subprocess.call(["mv", "script/temp_figs/screenshot_000.png", f"script/task_screenshots/{task_name}.png"])
    # cast all images to mp4
    end_time = time.time()
    frames2mp4(task_name+"_demo"+f"_setup2end{end_time-setup_begin_time:.3f}_execution2end{end_time-execution_begin_time:.3f}")

    # Ensure all frames are saved
    print("Task execution and frame saving completed.")
    time.sleep(5)


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

def eval_function_decorator(policy_name, model_name):
    """
    from ${policy_name} (like ACT or RDT) import a module named ${model_name}, which is an embodied model
    and create an FACTORY of this model
    """
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def main():
    task_files = ['adjust_bottle.py', 'dump_bin_bigbin.py', 'move_pillbottle_pad.py', 'place_bread_basket.py', 'place_mouse_pad.py', 'stack_bowls_two.py',
                  'move_playingcard_away.py', 'place_bread_skillet.py', 'place_object_basket.py', 'stamp_seal.py', 'beat_block_hammer.py', 'grab_roller.py',
                  'move_stapler_pad.py', 'place_burger_fries.py', 'place_object_scale.py', 'rotate_qrcode.py', 'turn_switch.py', 'blocks_ranking_rgb.py',
                  'handover_block.py', 'open_laptop.py', 'place_can_basket.py', 'place_object_stand.py', 'scan_object.py', 'blocks_ranking_size.py',
                  'handover_mic.py', 'open_microwave.py', 'place_cans_plasticbox.py', 'place_phone_stand.py', 'shake_bottle_horizontally.py', 'hanging_mug.py',
                  'pick_diverse_bottles.py', 'place_container_plate.py', 'place_shoe.py', 'shake_bottle.py', 'click_alarmclock.py', 'pick_dual_bottles.py',
                  'place_dual_shoes.py', 'press_stapler.py', 'stack_blocks_three.py', 'click_bell.py', 'lift_pot.py', 'place_a2b_left.py', 'place_empty_cup.py',
                  'put_bottles_dustbin.py', 'stack_blocks_two.py', 'move_can_pot.py', 'place_a2b_right.py', 'place_fan.py', 'put_object_cabinet.py', 'stack_bowls_three.py']

    task_names = [
        task_file.split('.')[0] for task_file in task_files
    ]

    for task_name in task_names:
        print(f'rendering {task_name}')
        render_env(task_name)

def render_env(task_name:str="beat_block_hammer"):
    # orig instruction:
    # bash eval.sh beat_block_hammer demo_randomized 0 50 0 0

    setup_begin_time = time.time()

    kwargs = {
        "task_name": task_name,
        "task_config": "demo_randomized",
        "ckpt_setting": 0,
        "expert_data_num": 50,
        "seed": 0,
        "gpu_id": 0,

        "policy_name":'ACT',

        # "DEBUG": False,
        "DEBUG": True,
    }
    kwargs["ckpt_dir"] = f"policy/ACT/act_ckpt/act-{kwargs['task_name']}/{kwargs['ckpt_setting']}-{kwargs['expert_data_num']}"
    kwargs['config'] = f"policy/{kwargs['policy_name']}/deploy_policy.yml"
    
    with open(kwargs['config'], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    config.update(kwargs)
    kwargs = config

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
    args['render_freq'] = 30 # force rendering
    print("args['render_freq']", args['render_freq'])

    args["eval_mode"] = True
    print("SET EVAL_MODE=TRUE")

    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
    # print(TASK_ENV.render_freq)
    # print(TASK_ENV.viewer)
    # print(TASK_ENV.viewer.render_scene)
    # print(TASK_ENV.viewer.scene)
    # # print(TASK_ENV.viewer.render_scene.update_render())
    # print(TASK_ENV.viewer.scene.update_render())

    # try to screen shot
    # viewer = TASK_ENV.viewer
    # print("press q to roll to next stage")
    
    # show demo
    # TASK_ENV.play_once()

    # meanwhile, take screenshots
    
    render_task(TASK_ENV, kwargs['task_name'], setup_begin_time=setup_begin_time)
    print("closing env")
    TASK_ENV.viewer.close()
    print("env closed")

if __name__ == '__main__':
    from test_render import Sapien_TEST
    Sapien_TEST()

    main()