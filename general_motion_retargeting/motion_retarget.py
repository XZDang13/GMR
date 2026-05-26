
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
        ik_mode: str="adaptive",
        max_iter: int=10,
        min_improvement: float=1e-3,
        task_weight_epsilon: float=0.0,
    ) -> None:
        if ik_mode not in ("adaptive", "single-pass"):
            raise ValueError(f"Unsupported ik_mode: {ik_mode}")
        self.verbose = verbose
        self.ik_mode = ik_mode
        self.max_iter = max(0, int(max_iter))
        self.min_improvement = float(min_improvement)
        self.task_weight_epsilon = max(0.0, float(task_weight_epsilon))

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        if verbose:
            print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        if verbose:
            print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        if verbose:
            print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.human_scale_anchor_table = ik_config.get("human_scale_anchor_table", {})
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}
        self.root_frame_pos_offsets = {
            key: np.array(value, dtype=float)
            for key, value in ik_config.get("root_frame_position_offsets", {}).items()
        }
        self.joint_regularization = {
            key: float(value)
            for key, value in ik_config.get("joint_regularization", {}).items()
        }
        self.posture_task = None

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        if self.verbose:
            for i in range(self.model.njnt):
                print(f"{i}: {self.model.joint(i).name}")
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if self.should_store_offset(pos_weight, rot_weight, pos_offset, rot_offset):
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
            if self.should_create_task(pos_weight, rot_weight):
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if self.should_store_offset(pos_weight, rot_weight, pos_offset, rot_offset):
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
            if self.should_create_task(pos_weight, rot_weight):
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.tasks2.append(task)
                self.task_errors2[task] = []

        self.setup_joint_regularization()

    def setup_joint_regularization(self):
        if not self.joint_regularization:
            return

        cost = np.zeros(self.model.nv)
        dof_counts = {
            int(mj.mjtJoint.mjJNT_FREE): 6,
            int(mj.mjtJoint.mjJNT_BALL): 3,
            int(mj.mjtJoint.mjJNT_SLIDE): 1,
            int(mj.mjtJoint.mjJNT_HINGE): 1,
        }
        for joint_name, joint_cost in self.joint_regularization.items():
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Unknown joint in joint_regularization: {joint_name}")
            dof_start = int(self.model.jnt_dofadr[joint_id])
            dof_count = dof_counts[int(self.model.jnt_type[joint_id])]
            cost[dof_start:dof_start + dof_count] = joint_cost

        self.posture_task = mink.PostureTask(self.model, cost=cost, lm_damping=1)
        self.posture_task.set_target(self.configuration.q.copy())
        self.tasks2.append(self.posture_task)

    def should_create_task(self, pos_weight, rot_weight):
        return (
            np.any(np.abs(np.atleast_1d(pos_weight).astype(float)) > self.task_weight_epsilon)
            or np.any(np.abs(np.atleast_1d(rot_weight).astype(float)) > self.task_weight_epsilon)
        )

    def should_store_offset(self, pos_weight, rot_weight, pos_offset, rot_offset):
        if (
            np.any(np.atleast_1d(pos_weight).astype(float) != 0.0)
            or np.any(np.atleast_1d(rot_weight).astype(float) != 0.0)
        ):
            return True
        pos_offset = np.asarray(pos_offset, dtype=float)
        rot_offset = np.asarray(rot_offset, dtype=float)
        return (
            np.linalg.norm(pos_offset) > 0.0
            or np.linalg.norm(rot_offset - np.array([1.0, 0.0, 0.0, 0.0])) > 0.0
        )

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(
            human_data,
            self.human_root_name,
            self.human_scale_table,
            self.human_scale_anchor_table,
        )
        human_data = self.offset_human_data(
            human_data,
            self.pos_offsets1,
            self.rot_offsets1,
            self.root_frame_pos_offsets,
        )
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            self.solve_task_set(self.tasks1, self.error1)

        if self.use_ik_match_table2:
            self.solve_task_set(self.tasks2, self.error2)
                
            
        return self.configuration.data.qpos.copy()

    def solve_task_set(self, tasks, error_func):
        if not tasks:
            return

        dt = self.configuration.model.opt.timestep
        if self.ik_mode == "single-pass" or self.max_iter == 0:
            self.solve_ik_once(tasks, dt)
            return

        curr_error = error_func()
        self.solve_ik_once(tasks, dt)
        next_error = error_func()
        num_iter = 0
        while curr_error - next_error > self.min_improvement and num_iter < self.max_iter:
            curr_error = next_error
            self.solve_ik_once(tasks, dt)
            next_error = error_func()
            num_iter += 1

    def solve_ik_once(self, tasks, dt):
        vel = mink.solve_ik(
            self.configuration, tasks, dt, self.solver, self.damping, self.ik_limits
        )
        self.configuration.integrate_inplace(vel, dt)

    def error1(self):
        if not self.tasks1:
            return 0.0
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        if not self.tasks2:
            return 0.0
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(
        self,
        human_data,
        human_root_name,
        human_scale_table,
        human_scale_anchor_table=None,
    ):
        human_scale_anchor_table = human_scale_anchor_table or {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        scaled_positions = {human_root_name: scaled_root_pos}

        def scaled_position(body_name, stack=None):
            if body_name in scaled_positions:
                return scaled_positions[body_name]
            if body_name not in human_data or body_name not in human_scale_table:
                raise KeyError(f"Cannot scale missing human body: {body_name}")
            stack = set() if stack is None else stack
            if body_name in stack:
                raise ValueError(f"Cycle in human_scale_anchor_table at {body_name}")
            stack.add(body_name)

            anchor_name = human_scale_anchor_table.get(body_name, human_root_name)
            if anchor_name == body_name:
                raise ValueError(f"Human body {body_name} cannot scale relative to itself")
            if anchor_name == human_root_name:
                anchor_raw_pos = root_pos
                anchor_scaled_pos = scaled_root_pos
            else:
                if anchor_name not in human_data:
                    raise KeyError(f"Scale anchor {anchor_name} for {body_name} is missing")
                anchor_raw_pos = human_data[anchor_name][0]
                anchor_scaled_pos = scaled_position(anchor_name, stack)

            scaled_positions[body_name] = (
                anchor_scaled_pos
                + (human_data[body_name][0] - anchor_raw_pos) * human_scale_table[body_name]
            )
            stack.remove(body_name)
            return scaled_positions[body_name]

        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data.keys():
            if body_name == human_root_name or body_name not in human_scale_table:
                continue
            human_data_global[body_name] = (scaled_position(body_name), human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets, root_frame_pos_offsets=None):
        """the pos offsets are applied in joint local frame; optional root offsets in human root frame"""
        offset_human_data = {}
        root_frame_pos_offsets = root_frame_pos_offsets or {}
        root_rotation = None
        if self.human_root_name in human_data and root_frame_pos_offsets:
            root_rotation = R.from_quat(human_data[self.human_root_name][1], scalar_first=True)

        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            if body_name not in pos_offsets or body_name not in rot_offsets:
                updated_quat = quat
            else:
                # apply rotation offset first
                updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
                offset_human_data[body_name][1] = updated_quat
            
                local_offset = pos_offsets[body_name]
                # compute the global position offset using the updated rotation
                global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
                
                offset_human_data[body_name][0] = pos + global_pos_offset

            if body_name in root_frame_pos_offsets and root_rotation is not None:
                root_pos_offset = root_rotation.apply(root_frame_pos_offsets[body_name])
                offset_human_data[body_name][0] = offset_human_data[body_name][0] + root_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
