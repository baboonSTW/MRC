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
import pybullet_utils.bullet_client as bc
import math
import sys
import numpy as np
import pybullet_data
import argparse
import time
from os.path import abspath, dirname, join
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from ConstrainedPlanningCommon import *

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import pb_ompl
# import ConstrainedPlanningCommon
# import ConstraintGeneration

class EEConstraint(ob.Constraint):
    def __init__(self, robot):
        super().__init__(7, 1)  # One constraint at a time
        self.robot = robot
        self.init_joint_state = self.robot.get_cur_state()
        self.init_ee_pose, self.init_ee_orientation = self.robot.get_ee_pose_from_state(self.init_joint_state)
        
    def function(self, x, out):
        current_ee_pose, _ = self.robot.get_ee_pose_from_state(x)
        # print(current_ee_pose)
        out[0] = abs(self.init_ee_pose[0] - current_ee_pose[0]) + abs(self.init_ee_pose[1] - current_ee_pose[1])

class DisplacementGoalRegion(ob.GoalRegion):
    def __init__(self, si, robot):
        super().__init__(si)
        self.robot = robot
        # Make sure you run self.robot.set_state(state) before create the goal instance
        self.init_joint_state = self.robot.get_cur_state()
        self.init_ee_pose, self.init_ee_orientation = self.robot.get_ee_pose_from_state(self.init_joint_state)
        self.goal_ee_pose = self.init_ee_pose - np.array([0,0,0.4])
        
        self.threshold = None
        self.setThreshold(0.001)
        
        # FOR DEBUGGING
        self.closest = np.inf
        print("Init ee pose: ", self.init_ee_pose)
        print("Goal ee pose: ", self.goal_ee_pose)
        
    def distanceGoal(self, state):
        current_ee_pose, _ = self.robot.get_ee_pose_from_state(state)
        dis = np.linalg.norm(current_ee_pose-self.goal_ee_pose)
        print(dis)
        return dis
    
    def isSatisfied(self, state, dis):
        current_ee_pose, _ = self.robot.get_ee_pose_from_state(state)
        dis = np.linalg.norm(current_ee_pose-self.goal_ee_pose)
        if dis < self.closest:
            self.closest = dis
            print("closest: ", self.closest)
        threshold = self.getThreshold()
        if dis < threshold:
            return True
        return False 
           
    def getThreshold(self):
        return self.threshold
    
    def setThreshold(self, threshold):
        self.threshold = threshold
        
    # def isSatisfied(self, state):
    #     current_ee_pose, _ = self.robot.get_ee_pose_from_state(state)
    #     dis = np.linalg.norm(current_ee_pose-self.goal_ee_pose)
    #     if dis < self.closest:
    #         self.closest = dis
    #         print("In origin satisfied function. closest: ", self.closest)
    #     if dis < 0.01:
    #         return True
    #     return False 
    
class DisplacementGoal(ob.Goal):
    def __init__(self, si, robot):
        super().__init__(si)
        self.robot = robot
        # Make sure you run self.robot.set_state(state) before create the goal instance
        self.init_joint_state = self.robot.get_cur_state()
        self.init_ee_pose, self.init_ee_orientation = self.robot.get_ee_pose_from_state(self.init_joint_state)
        self.goal_ee_pose = self.init_ee_pose - np.array([0,0,0.4])
        # self.threshold = None
        # self.setThreshold(0.001)
        
        # FOR DEBUGGING
        self.closest = np.inf
        print("Init ee pose: ", self.init_ee_pose)
        print("Goal ee pose: ", self.goal_ee_pose)
        
    def isSatisfied(self, state):
        current_ee_pose, _ = self.robot.get_ee_pose_from_state(state)
        dis = np.linalg.norm(current_ee_pose-self.goal_ee_pose)
        if dis < self.closest:
            self.closest = dis
            print("In origin satisfied function. closest: ", self.closest)
        if dis < 0.001:
            return True
        return False 
           
class Demo():
    
    def planningOnce(self, cp, plannername, output):
        cp.setPlanner(plannername, "plan_on_constraint")
    
        # Solve the problem
        stat = cp.solveOnce(output, "plan_on_constraint")
    
        if output:
            ou.OMPL_INFORM("Dumping problem information to `plan_on_constraint.txt`.")
            with open("sphere_info.txt", "w") as infofile:
                print(cp.spaceType, file=infofile)
    
        cp.atlasStats()
        if output:
            cp.dumpGraph("plan_on_constraint")
        return stat
  
    def planningBench(self, cp, planners):
        cp.setupBenchmark(planners, "plan_on_constraint")
        cp.runBenchmark()
        
    def constraintPlanning(self, robot, start_joints, options):
        # Create the ambient space state space for the problem 
        print("====\nNumber Joint: ", robot.num_dim)
        self.space = ob.RealVectorStateSpace(robot.num_dim)
        bounds = ob.RealVectorBounds(robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)
        
        # Create the constraints
        constraint = EEConstraint(self.robot)
        
        self.cp = ConstrainedProblem(options.space, self.space, constraint, options)
        
        # cp.css.registerProjection("sphere", SphereProjection(cp.css))
        
        start = ob.State(self.cp.css)
        # goal = ob.State(cp.css)
        goal = DisplacementGoal(self.cp.csi, self.robot)
        # print("Start state: ", dir(start))
        # print("Start state before: ", start)

        for i in range(len(start_joints)):
            start[i] = start_joints[i]
            # goal[i] = goal_joints[i]
        # print("Start state after: ", start)  
        # Set start and goal 
        self.cp.setStartAndGoalStates(start, goal)
        
        # cp.ss.setStateValidityChecker(ob.StateValidityCheckerFn(obstacles))
        
        self.planners = options.planner.split(",")
        # if not options.bench:
        #     self.planningOnce(cp, self.planners[0], options.output)
        # else:
        #     self.planningBench(cp, self.planners)

    def __init__(self):
        self.obstacles = []

        # Connect to PyBullet in GUI mode for visualization
        self.visual_bullet_connection = bc.BulletClient(connection_mode=p.GUI)
        # Connect to PyBullet in DIRECT mode for computation
        self.visual_bullet_connection.setGravity(0, 0, -9.8)
        self.visual_bullet_connection.setTimeStep(1./240.)

        self.visual_bullet_connection.setAdditionalSearchPath(pybullet_data.getDataPath())

        # p.loadURDF("plane.urdf")
        
        # load robot
        visual_robot_id = self.visual_bullet_connection.loadURDF("franka_panda/panda_with_stick.urdf", [2.5, -0.5, 3], [0, 1, 0, 0], useFixedBase = 1)
        robot = pb_ompl.PbOMPLRobot(visual_robot_id, self.visual_bullet_connection)
        self.robot = robot
        # add obstacles
        # self.add_obstacles()
        
    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]
    
    def pybullet_execute(self, path, dynamics=False):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    self.visual_bullet_connection.setJointMotorControl2(self.robot.id_visual, i, self.visual_bullet_connection.POSITION_CONTROL, q[i],force=5 * 240.)
            else:
                self.robot.set_state(q)
            self.visual_bullet_connection.stepSimulation()
            time.sleep(0.01)
            
    def demo(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("-o", "--output", action="store_true",
                         help="Dump found solution path (if one exists) in plain text and planning "
                         "graph in GraphML to `sphere_path.txt` and `sphere_graph.graphml` "
                         "respectively.")
        parser.add_argument("--bench", action="store_true",
                         help="Do benchmarking on provided planner list.")
        addSpaceOption(parser)
        addPlannerOption(parser)
        addConstrainedOptions(parser)
        addAtlasOptions(parser)
        
        start_joints = [0,0,0,-1,0,1.5,0]
        self.robot.set_state(start_joints)
        
        goal = [1,1,0,-1,0,1.5,0]
        
        print("====\n", parser.parse_args())
        self.constraintPlanning(self.robot, start_joints, parser.parse_args())
        
        # self.constraint.set_start_state(start)
        # self.constraint.set_movement_constraint()
        
        stat = None
        if not parser.parse_args().bench:
            stat, simplePath = self.planningOnce(self.cp, self.planners[0], parser.parse_args().output)
            if stat:
                sol_path_states = simplePath.getStates()
                sol_path_list = [self.state_to_list(state) for state in sol_path_states]
                print(sol_path_list)
                while True:
                    self.pybullet_execute(sol_path_list)
        else:
            stat = self.planningBench(self.cp, self.planners)
  
        # goal = DisplacementGoal(self.pb_ompl_interface.si, self.robot.id, 11, start, goal_displacement)
        
        # res, path = self.pb_ompl_interface.plan(goal)
        # if res:
        #     self.pb_ompl_interface.execute(path)
        # return res, path

if __name__ == '__main__':
    env = Demo()
    env.demo()