from ._base_task import Base_Task
from .utils import *
import sapien
import math


class PickAndPlaceObject(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Randomly select a pose for the object
        self.qpose_tag = np.random.randint(0, 2)
        qposes = [[0.707, 0.0, 0.0, -0.707], [0.707, 0.0, 0.0, 0.707]]
        xlims = [[-0.12, -0.08], [0.08, 0.12]]

        # Load the object to be picked and placed
        self.model_id = np.random.choice([13, 16])  # Example model IDs

        self.object = rand_create_actor(
            self,
            xlim=xlims[self.qpose_tag],
            ylim=[-0.13, -0.08],
            zlim=[0.752],
            rotate_rand=True,
            qpos=qposes[self.qpose_tag],
            modelname="001_bottle",  # Example model name
            convex=True,
            rotate_lim=(0, 0, 0.4),
            model_id=self.model_id,
        )
        self.delay(4)
        self.add_prohibit_area(self.object, padding=0.15)
        self.left_target_pose = [-0.25, -0.12, 0.95, 0, 1, 0, 0]
        self.right_target_pose = [0.25, -0.12, 0.95, 0, 1, 0, 0]

    def play_once(self):
        # Determine which arm to use based on object's position
        arm_tag = ArmTag("right" if self.qpose_tag == 1 else "left")
        # Select the target pose based on object's initial position
        target_pose = (self.right_target_pose if self.qpose_tag == 1 else self.left_target_pose)

        # Grasp the object with the specified arm
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        # Move the arm upward by 0.1 meters along z-axis
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        # Place the object at the target pose
        self.move(
            self.place_actor(
                self.object,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                is_open=False,
            ))

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        target_height = 0.9
        object_pose = self.object.get_functional_point(0)
        return ((self.qpose_tag == 0 and object_pose[0] < -0.15) or
                (self.qpose_tag == 1 and object_pose[0] > 0.15)) and object_pose[2] > target_height
