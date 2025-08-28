# import packages and module here

#导包
from typing import Any, Dict
import numpy as np

#模型类
class HelloModel:
    def __init__(self, usr_args: Dict):
        self.dz = float(usr_args.get("dz", 0.05))
        self.open_close = bool(usr_args.get("open_close", True))
        self.steps = int(usr_args.get("steps", 25))
        self.t = 0
    def reset(self):
        self.t = 0

# 不处理
# 尝试-> dict
def encode_obs(observation)-> dict:  # Post-Process Observation
    obs = observation
    # ...
    return obs

# #不加载
# #尝试-> Any
# def get_model(usr_args)-> Any:  # from deploy_policy.yml and eval.sh (overrides)
#     Your_Model = None
#     # ...
#     return Your_Model  # return your policy model

#新的get_model，前面的报错
def get_model(usr_args: dict) -> Any:
    print("[HelloPolicy] get_model called with:", usr_args)
    return HelloModel(usr_args)


#签名，新增【:HelloModel】，【:dict】，【-> Any】
def eval(TASK_ENV, model:HelloModel, observation:dict)-> Any:

    #警告，保持逻辑
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """

    # 新增，因为报错说model是None
    # 关键补丁：若上层传入 None（如 --ckpt_setting none），这里自行构建一个默认模型
    if model is None:
        model = get_model({"dz": 0.05, "open_close": True, "steps": 25})

    obs = encode_obs(observation)  # Post-Process Observation
    
    # 注释掉，没用到
    # instruction = TASK_ENV.get_instruction()

    # 新增。调用获取动作，更新观测
    for _ in range(model.steps):
        action = get_action(model, obs)
        TASK_ENV.take_action(action, action_type="ee")  # 关键调用：按 ee 控制
        # 如果环境支持取最新观测，就更新一下
        if hasattr(TASK_ENV, "get_observation"):
            obs = encode_obs(TASK_ENV.get_observation())
    return {"ok": True}


    # 下面是用于policy的，这里不用

    # if len(
    #         model.obs_cache
    # ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
    #     model.update_obs(obs)

    # actions = model.get_action()  # Get Action according to observation chunk

    # for action in actions:  # Execute each step of the action
    #     # see for https://robotwin-platform.github.io/doc/control-robot.md more details
    #     TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
    #     # TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
    #     observation = TASK_ENV.get_obs()
    #     obs = encode_obs(observation)
    #     model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


# 尝试-> None
def reset_model(model)-> None:  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass

# 新增的，不过没用
def update_obs(obs: dict) -> None:
    pass

# 新增
def get_action(model: HelloModel, obs: dict) -> np.ndarray:
    """
    生成一次 ee 动作：
    - 左臂：Z 轴上抬 dz
    - 右臂：保持原位
    - 夹爪：左臂前半程关闭、后半程打开；右臂始终打开
    ee 动作格式：[left_xyzquat(7), left_gripper, right_xyzquat(7), right_gripper]
    """
    ep = obs.get("endpose", {})
    L = list(ep.get("left_endpose",  [0,0,0, 1,0,0,0]))
    R = list(ep.get("right_endpose", [0,0,0, 1,0,0,0]))
    lg = float(ep.get("left_gripper",  1.0))
    rg = float(ep.get("right_gripper", 1.0))

    # 左臂上抬
    L[2] = L[2] + model.dz

    # 简单的开合节奏（只演示）
    phase = (model.t / max(1, model.steps))
    if model.open_close:
        lg = 0.0 if phase < 0.5 else 1.0   # 前半程闭合，后半程打开
        rg = 1.0                           # 右爪保持张开

    action = np.array(L + [lg] + R + [rg], dtype=np.float32)
    model.t += 1
    return action