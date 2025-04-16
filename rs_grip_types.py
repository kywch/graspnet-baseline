import time
import json
import numpy as np

import robosuite
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.inspire_hands import InspireRightHand


class CustomInspireRightHand(InspireRightHand):
    def __init__(self, idn=0):
        super().__init__(idn)  # Use Rososuite's inspire right hand xml

        # NOTE: where to put this json file?
        with open("inspire_width_angle.json", "r", encoding="utf-8") as f:
            self._width_angle_dict = json.load(f)

        # Get control range
        ctrl_range = np.array(
            [[float(x) for x in ac.attrib["ctrlrange"].split(" ")] for ac in self._elements["actuators"]]
        )
        self.control_range = (ctrl_range[:, 0] - ctrl_range[:, 1]) / 1000
        self.control_base = ctrl_range[:, 1]

        # Convert 6d to 12d: from robosuite/models/grippers/inspire_hands.py, InspireRightHand.format_action()
        self._map_6d_to_12d = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])

        # Convert the json file finger pos (0, 1000) to the 12-dof control signal
        self.grip_info = {}
        for grip_type in self._width_angle_dict:
            self.grip_info[grip_type] = {"valid_widths": []}

            for width in self._width_angle_dict[grip_type]:
                # if 6d is -1, then it's not valid
                action_6d = self._width_angle_dict[grip_type][width]["6d"]
                if action_6d[0] > -1:
                    self.grip_info[grip_type]["valid_widths"].append(width)
                    self._width_angle_dict[grip_type][width]["12d"] = self._convert_6d_to_12d(action_6d)

            self.grip_info[grip_type]["idx_scale"] = (len(self.grip_info[grip_type]["valid_widths"]) - 1) / 2.0

        # Set the gripper type
        self._grip_type = "Tripod"
        assert self._grip_type in self.grip_info, "Gripper type {} not found in gripper info!".format(self._grip_type)

        print()

    def _convert_6d_to_12d(self, action_6d):
        return np.array(action_6d)[self._map_6d_to_12d] * self.control_range + self.control_base

    def set_grip_type(self, grip_type):
        assert grip_type in self.grip_info, "Gripper type {} not found in gripper info!".format(grip_type)
        self._grip_type = grip_type

    def format_action(self, action):
        assert len(action) == self.dof, "Action dimension {} does not match the gripper dof {}".format(
            len(action), self.dof
        )
        assert -1 <= action[0] <= 1, "Action value {} is not in [-1, 1]".format(action)

        action_to_idx = int((action[0] + 1) * self.grip_info[self._grip_type]["idx_scale"])
        width_key = self.grip_info[self._grip_type]["valid_widths"][action_to_idx]

        return self._width_angle_dict[self._grip_type][width_key]["12d"]

    @property
    def dof(self):
        return 1


register_gripper(CustomInspireRightHand)

robot = "PandaDexRH"

# NOTE: consider using mink or curobo along with the default OSC delta controller
env = robosuite.make(
    "Lift",
    robots=[robot],
    # controller_configs=controller_config,
    gripper_types="CustomInspireRightHand",
    has_renderer=True,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_object_obs=True,
    use_camera_obs=True,
    control_freq=20,
)

env.reset()

gripper = env.robots[0].gripper["right"]
# -1 is closed, 1 is open
gripper_seq = [-1.0] * 10 + list(np.arange(-1, 1, 0.042)) + [1.0] * 2 + list(np.arange(1, -1, -0.042))

# See env.robots[0].part_controllers for arm and hand control
eef_pose = np.zeros(7)  # delta controller

for grip_type in list(gripper.grip_info.keys()):
    gripper.set_grip_type(grip_type)
    print("Playing grip:", grip_type)

    for _ in range(3):
        for i in range(len(gripper_seq)):
            eef_pose[-1] = gripper_seq[i]
            env.step(eef_pose)
            time.sleep(0.02)

print("Done.")
