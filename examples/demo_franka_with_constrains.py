try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
    # sys.path.insert(0, join(dirname(abspath(__file__)), '../whole-body-motion-planning/src/ompl/py-bindings'))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
import os.path as osp
import pybullet as p
import math
import sys
import numpy as np
import pybullet_data
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import pb_ompl
# import ConstrainedPlanningCommon
# import ConstraintGeneration

class EEConstraint(ob.Constraint):
    def __init__(self, robot_id):
        super().__init__(7, 2)  # One constraint at a time
        self.robot_id = robot_id
        self.rotation_point = None
        self.axis_direction = None
        self.movement_direction = None
        self.start_state = None
        self.start_pos = None
        self.start_rotation_matrix = None
        self.ee_index = 11
        
    def set_start_state(self, start_state):
        self.start_state = start_state
        
    def set_rotational_constraint(self, axis_point, axis_direction):
        self.rotation_point = np.array(axis_point)
        self.axis_direction = np.array(axis_direction) / np.linalg.norm(axis_direction)  # Normalize the direction
        self.movement_direction = None  # Disable movement constraint
        # Capture the initial orientation of the end effector with respect to the screw axis
        _, start_quat = self.get_ee_pose_from_state(self.robot_id, self.ee_index, self.start_state)
        self.start_rotation_matrix = p.getMatrixFromQuaternion(start_quat)
        self.start_rotation_matrix = np.array(self.start_rotation_matrix).reshape((3, 3))

    def set_movement_constraint(self):
        _, start_quat = self.get_ee_pose_from_state(self.robot_id, self.ee_index, self.start_state)
        self.movement_direction = self.quat_to_direction(start_quat)
        self.rotation_point = None  # Disable rotational constraint
        self.start_pos, _ = self.get_ee_pose_from_state(self.robot_id, self.ee_index, self.start_state)  # Store start_pos

    def function(self, x, out):
        pos, quat = self.get_ee_pose_from_state(self.robot_id, self.ee_index, x)
        
        if self.rotation_point is not None and self.axis_direction is not None:
            # Rotational Constraint
            vector_to_ee = pos - self.rotation_point
            cross_product = np.cross(self.axis_direction, vector_to_ee)
            out[0] = np.linalg.norm(cross_product)

            # Orientation Constraint
            current_rotation_matrix = p.getMatrixFromQuaternion(quat)
            current_rotation_matrix = np.array(current_rotation_matrix).reshape((3, 3))
            rotation_error_matrix = np.dot(current_rotation_matrix, self.start_rotation_matrix.T)
            identity_matrix = np.eye(3)
            rotation_error = np.linalg.norm(rotation_error_matrix - identity_matrix)
            out[1] = rotation_error
        
        elif self.movement_direction is not None:
            # Directional Movement Constraint
            current_direction = pos - self.start_pos  # Use the stored start_pos
            current_direction = current_direction / np.linalg.norm(current_direction)
            print("in constraint Movement")
            out[0] = np.dot(self.movement_direction, current_direction) - 1  # Should be 1 if they are aligned
            out[1] = 0.0
        else:
            print("in constraint None")
            out[0] = 0.0  # No constraint
            out[1] = 0.0

    def get_ee_pose_from_state(self, robot_id, ee_index, state):
        for j in range(7):
            p.resetJointState(robot_id, j, state[j])
        ee_state = p.getLinkState(robot_id, linkIndex=ee_index)
        pos, quat = np.array(ee_state[4]), np.array(ee_state[5])
        return pos, quat

    def quat_to_direction(self, quat):
        # Convert quaternion to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(quat)
        rot_matrix = np.array(rot_matrix).reshape((3, 3))
        # Assuming the forward direction corresponds to the x-axis of the end effector's local frame
        forward_direction = rot_matrix[:, 0]  
        return forward_direction

class DisplacementGoal(ob.Goal):
    def __init__(self, space, robot_id, ee_index, start_state, displacement):
        super().__init__(space)
        self.robot_id = robot_id
        self.ee_index = ee_index
        self.start_state = start_state
        self.displacement = displacement
        self.start_pos, _ = self.get_ee_pose_from_state(robot_id, ee_index, start_state)

    def isSatisfied(self, state):
        current_pos, _ = self.get_ee_pose_from_state(self.robot_id, self.ee_index, state)
        current_displacement = np.linalg.norm(current_pos - self.start_pos)
        return np.isclose(current_displacement, self.displacement, atol=1e-3), 0.0  # Tolerance of 1e-3

    def get_ee_pose_from_state(self, robot_id, ee_index, state):
        for j in range(7):
            p.resetJointState(robot_id, j, state[j])
        ee_state = p.getLinkState(robot_id, linkIndex=ee_index)
        pos, quat = np.array(ee_state[4]), np.array(ee_state[5])
        return pos, quat
           
class Demo():
    def __init__(self):
        self.obstacles = []

        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # load robot
        robot_id = p.loadURDF("franka_panda/panda_with_stick.urdf", [2.5, -0.5, 3], [0, 1, 0, 0], useFixedBase = 1)
        robot = pb_ompl.PbOMPLRobot(robot_id)
        self.robot = robot
        
        # setup constrain
        self.constraint = EEConstraint(self.robot.id)
        
        
        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.constraint, self.obstacles)
        self.pb_ompl_interface.set_planner("RRT")

        # add obstacles
        self.add_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_obstacles(self):
        # add box
        self.add_box([1, 0, 0.7], [0.5, 0.5, 0.05])

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)
        return box_id

    def demo(self):
        start = [0,0,0,-1,0,1.5,0]
        goal_displacement = 0.4

        self.robot.set_state(start)
        self.constraint.set_start_state(start)
        self.constraint.set_movement_constraint()
        
        goal = DisplacementGoal(self.pb_ompl_interface.si, self.robot.id, 11, start, goal_displacement)
        
        res, path = self.pb_ompl_interface.plan(goal)
        if res:
            self.pb_ompl_interface.execute(path)
        return res, path

if __name__ == '__main__':
    env = Demo()
    env.demo()