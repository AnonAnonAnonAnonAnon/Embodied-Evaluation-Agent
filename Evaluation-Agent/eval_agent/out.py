from pathlib import Path
from typing import List, Dict, Union

def dump_py_files(items: Union[List[Dict[str, str]], Dict[str, str]],
                  out_dir: Union[str, Path] = ".") -> List[Path]:
    """
    将形如 [{'a.py':'content'}, {'b.py':'content'}] 或 {'a.py':'content', ...}
    的结构写入到 out_dir 目录下，返回写入的文件路径列表。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统一成 list[dict]
    if isinstance(items, dict):
        dicts = [items]
    else:
        dicts = items

    written: List[Path] = []
    for d in dicts:
        for filename, content in d.items():
            # 防呆：确保扩展名
            if not str(filename).endswith(".py"):
                filename = f"{filename}.py"
            path = out_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            # 写入（确保末尾换行，习惯性）
            if content and not content.endswith("\n"):
                content = content + "\n"
            path.write_text(content, encoding="utf-8")
            written.append(path.resolve())
    return written

# 使用示例
if __name__ == "__main__":
    data = [{
        'task_place_cube_on_spot.py': 'from ._base_task import Base_Task\nfrom .utils import *\nimport sapien\n\n\nclass place_cube_on_spot(Base_Task):\n\n    def setup_demo(self, **kwags):\n        super()._init_task_env_(**kwags)\n\n    def load_actors(self):\n        # Randomly determine the position to place the cube on the table\n        spot_position = rand_pose(\n            xlim=[-0.15, 0.15],\n            ylim=[-0.1, 0.1],\n            zlim=[0.76],\n            qpos=[1, 0, 0, 0],\n            rotate_rand=True,\n            rotate_lim=[0, 0, 0.5],\n        )\n        while abs(spot_position.p[0]) < 0.05:\n            spot_position = rand_pose(\n                xlim=[-0.15, 0.15],\n                ylim=[-0.1, 0.1],\n                zlim=[0.76],\n                qpos=[1, 0, 0, 0],\n                rotate_rand=True,\n                rotate_lim=[0, 0, 0.5],\n            )\n\n        self.cube = create_box(\n            scene=self,\n            pose=rand_pose(xlim=[-0.25, 0.25], ylim=[-0.15, 0.15], zlim=[0.76], qpos=[1, 0, 0, 0]),\n            half_size=(0.03, 0.03, 0.03),\n            color=(0, 0, 1),  # Blue cube\n            name=\"cube\",\n            is_static=False,\n        )\n\n        self.spot = create_box(\n            scene=self,\n            pose=spot_position,\n            half_size=(0.03, 0.03, 0.01),  # Smaller height for spot\n            color=(0, 1, 0),  # Green spot\n            name=\"spot\",\n            is_static=True,\n        )\n\n        self.add_prohibit_area(self.cube, padding=0.05)\n\n    def play_once(self):\n        # Determine which arm to use based on cube position (left if cube is on the left side, else right)\n        arm_tag = ArmTag(\"left\" if self.cube.get_pose().p[0] < 0 else \"right\")\n\n        # Grasp the cube with the selected arm\n        self.move(self.grasp_actor(self.cube, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))\n\n        # Move the cube upwards\n        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis=\"arm\"))\n\n        # Place the cube on the spot\n        self.move(\n            self.place_actor(\n                self.cube,\n                target_pose=self.spot.get_pose(),\n                arm_tag=arm_tag,\n                functional_point_id=0,\n                pre_dis=0.06,\n                dis=0,\n                is_open=False,\n            ))\n\n        self.info[\"info\"] = {\"{A}\": \"cube/base0\", \"{a}\": str(arm_tag)}\n        return self.info\n\n    def check_success(self):\n        cube_target_pose = self.cube.get_functional_point(0, \"pose\").p\n        spot_pose = self.spot.get_pose().p\n        eps = np.array([0.02, 0.02])\n        return np.all(abs(cube_target_pose[:2] - spot_pose[:2]) < eps)\n'
    }]
    paths = dump_py_files(data, out_dir="./output")
    print("Written:", *map(str, paths), sep="\n- ")
