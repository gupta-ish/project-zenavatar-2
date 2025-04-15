import numpy as np
import pinocchio as pin

class EE_MotionPlanner:
    def __init__(self, approach_offset=0.1, lift_height=0.2):
        self.approach_offset = approach_offset
        self.lift_height = lift_height

    def compute_wrist_yaw(self, tote_pose):
        tote_yaw = np.arctan2(tote_pose.translation[1], tote_pose.translation[0])
        wrist_yaw = {
            "left": tote_yaw - np.pi / 2, 
            "right": tote_yaw + np.pi / 2 
        }
        return wrist_yaw

    def generate_waypoints(self, tote_pose):
        tote_pos = tote_pose.translation
        tote_rot = tote_pose.rotation

        approach_pos = tote_pos + np.array([0, 0, self.approach_offset])
        grasp_pos = tote_pos.copy()
        lift_pos = tote_pos + np.array([0, 0, self.lift_height])

        waypoints = {
            "left": [
                pin.SE3(tote_rot, approach_pos + np.array([0.1, 0.2, 0])),  # Approach
                pin.SE3(tote_rot, grasp_pos + np.array([0.1, 0.2, 0])),    # Grasp
                pin.SE3(tote_rot, lift_pos + np.array([0.1, 0.2, 0]))      # Lift
            ],
            "right": [
                pin.SE3(tote_rot, approach_pos + np.array([-0.1, -0.2, 0])),
                pin.SE3(tote_rot, grasp_pos + np.array([-0.1, -0.2, 0])),
                pin.SE3(tote_rot, lift_pos + np.array([-0.1, -0.2, 0]))
            ]
        }

        # Compute wrist yaw
        wrist_yaw = self.compute_wrist_yaw(tote_pose)

        return waypoints, wrist_yaw
