import numpy as np
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from typing import Dict
import random

# env related
from envs.base_task import BaseTask

# utils
from utils.terrain import Terrain
from utils.math import quat_apply_yaw, wrap_to_pi, get_scale_shift
from utils.helpers import class_to_dict
import torchvision
import cv2

# config
from configs import LeggedRobotCfg
from global_config import ROOT_DIR
from utils.utils import random_quat

def quat_from_euler_xyz_local(roll, pitch, yaw):
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    quat = torch.zeros(roll.shape[0], 4, device=roll.device)
    quat[:, 0] = sr * cp * cy - cr * sp * sy
    quat[:, 1] = cr * sp * cy + sr * cp * sy
    quat[:, 2] = cr * cp * sy - sr * sp * cy
    quat[:, 3] = cr * cp * cy + sr * sp * sy
    return quat

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.global_counter = 0
    
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self._prepare_cost_function()
        self.init_done = True

        # self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # self.post_physics_step()

    #------------ enviorment core ----------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
            isaac gym order:
                0 FL_hip_joint 3
                1 FL_thigh_joint 4
                2 FL_calf_joint 5
                3 FR_hip_joint 0 
                4 FR_thigh_joint 1
                5 FR_calf_joint 2 
                6 RL_hip_joint 9 
                7 RL_thigh_joint 10
                8 RL_calf_joint 11
                9 RR_hip_joint 6 
                10 RR_thigh_joint 7
                11 RR_calf_joint 8
            unitree go2 sdk order:
                3 FR_hip_joint 0
                4 FR_thigh_joint 1
                5 FR_calf_joint 2
                0 FL_hip_joint 3
                1 FL_thigh_joint 4
                2 FL_calf_joint 5
                9 RR_hip_joint 6
                10 RR_thigh_joint 7
                11 RR_calf_joint 8
                6 RL_hip_joint 9
                7 RL_thigh_joint 10
                8 RL_calf_joint 11
        """
  
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
   
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        str_rng = self.cfg.domain_rand.motor_strength_range
        kp_str_rng = self.cfg.domain_rand.kp_range
        kd_str_rng = self.cfg.domain_rand.kd_range

        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        self.kp_factor = (kp_str_rng[1] - kp_str_rng[0]) * torch.rand(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + kp_str_rng[0]
        self.kd_factor = (kd_str_rng[1] - kd_str_rng[0]) * torch.rand(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + kd_str_rng[0]

        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)


        if self.cfg.env.history_encoding:
             self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.num_dofs, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.recovery_commands = torch.zeros_like(self.commands)
        self.recovery_command_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.recovery_upside_down_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.recovery_exact_upside_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
            requires_grad=False
        )
        self.recovery_left_side_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.recovery_right_side_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.recovery_last_gravity_z = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.recovery_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.recovery_success_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        # Latch whether strict success was reached at least once in the current
        # episode. Needed when recovery success is measured without terminating
        # the episode immediately.
        self.recovery_episode_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
            requires_grad=False
        )
        # A retry is counted only after the robot has escaped the feet-up basin
        # and subsequently falls back into it.  This distinguishes a decisive
        # first roll from a policy that eventually succeeds after rocking.
        self.recovery_retry_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
            requires_grad=False
        )
        self.recovery_reached_side = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
            requires_grad=False
        )
        self.recovery_first_success_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device,
            requires_grad=False
        )
        self.recovery_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
            requires_grad=False
        )
        # Environment-side state for the success-gated recovery curriculum.
        # It intentionally starts from stage zero for every scratch run.
        self.recovery_curriculum_stage = 0
        self.recovery_curriculum_success_ema = 0.0
        self.recovery_curriculum_stage_start_step = 0
        self.recovery_episode_min_leg_clearance = torch.full(
            (self.num_envs, 4), float("inf"),
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
      
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.base_height_points = self._init_base_height_points()

        self.measured_heights = 0
        self.feet_heights = 0
        self.feet_local_heights = torch.zeros(self.num_envs,12,dtype=torch.float, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_start_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            start_angle = self.cfg.init_state.start_joint_angles[name]

            self.default_dof_pos[i] = angle
            self.default_start_pos[i] = start_angle

            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_start_pos = self.default_start_pos.unsqueeze(0)
        self.recovery_curl_dof_pos = self.default_dof_pos.clone()
        self.recovery_stand_dof_pos = self.default_dof_pos.clone()
        for i, name in enumerate(self.dof_names):
            if "hip_joint" in name:
                self.recovery_curl_dof_pos[0, i] = getattr(
                    self.cfg.domain_rand, "recovery_curl_hip", 0.0
                )
                self.recovery_stand_dof_pos[0, i] = getattr(
                    self.cfg.domain_rand, "recovery_stand_hip", 0.0
                )
            elif "thigh_joint" in name:
                self.recovery_curl_dof_pos[0, i] = getattr(
                    self.cfg.domain_rand, "recovery_curl_thigh", 1.5
                )
                self.recovery_stand_dof_pos[0, i] = getattr(
                    self.cfg.domain_rand, "recovery_stand_thigh", 0.8
                )
            elif "calf_joint" in name:
                self.recovery_curl_dof_pos[0, i] = getattr(
                    self.cfg.domain_rand, "recovery_curl_calf", -2.4
                )
                self.recovery_stand_dof_pos[0, i] = getattr(
                    self.cfg.domain_rand, "recovery_stand_calf", -1.5
                )
        self.recovery_hip_dof_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "hip_joint" in name],
            dtype=torch.long, device=self.device
        )
        self.recovery_thigh_dof_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "thigh_joint" in name],
            dtype=torch.long, device=self.device
        )
        self.recovery_calf_dof_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "calf_joint" in name],
            dtype=torch.long, device=self.device
        )

        # Centerline samples derived from panda3_2.urdf collision primitives.
        # Link-origin distance alone is inaccurate for these long folded links.
        # About 2 cm spacing prevents the previous sparse proxy from missing
        # contact between sample points.
        calf_t = torch.linspace(0., 1., 11, device=self.device).unsqueeze(1)
        calf_start = to_torch([0.016, 0.0, 0.030], device=self.device)
        calf_end = to_torch([-0.008, 0.0, -0.280], device=self.device)
        self.recovery_calf_local_points = (
            calf_start + calf_t * (calf_end - calf_start)
        )
        cylinder_t = torch.linspace(0., 1., 9, device=self.device)
        box_t = torch.linspace(0., 1., 9, device=self.device)
        thigh_points = []
        for cylinder_y_start, cylinder_y_end in ((-0.136, 0.024), (-0.024, 0.136)):
            cylinder_points = torch.zeros(9, 3, device=self.device)
            cylinder_points[:, 1] = (
                cylinder_y_start
                + cylinder_t * (cylinder_y_end - cylinder_y_start)
            )
            box_points = torch.zeros(9, 3, device=self.device)
            box_points[:, 0] = -0.019
            box_points[:, 2] = -0.055 + box_t * (-0.275 + 0.055)
            thigh_points.append(torch.cat((cylinder_points, box_points), dim=0))
        # FL/FR/RL/RR use the matching left/right thigh geometry.
        self.recovery_thigh_local_points = torch.stack((
            thigh_points[0], thigh_points[1], thigh_points[0], thigh_points[1]
        ), dim=0)
        # Conservative cross-section radii convert sampled centerline distance
        # into an approximate collision-surface clearance.
        thigh_radii = torch.cat((
            torch.full((9,), 0.055, device=self.device),
            torch.full((9,), 0.036, device=self.device),
        ))
        calf_radii = torch.full((11,), 0.036, device=self.device)
        self.recovery_leg_point_radii = torch.cat(
            (thigh_radii, calf_radii)
        ).view(1, 1, -1)
        # Only front/rear pairs are relevant to the observed deadlock. The
        # same-end FL/FR and RL/RR hip cylinders are naturally close in the
        # nominal mechanism and must not create a permanent penalty.
        self.recovery_leg_pair_indices = torch.tensor(
            [[0, 2], [0, 3], [1, 2], [1, 3]],
            dtype=torch.long, device=self.device
        )

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)
            
        self.lag_buffer = torch.zeros(self.num_envs,self.cfg.domain_rand.lag_timesteps,self.num_actions,device=self.device,requires_grad=False)

        #phase related
        self.phase = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.phase_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.frequency = 2.
        
        self.trot_gait = torch.zeros(1, 4, dtype=torch.float, device=self.device,requires_grad=False)
        self.trot_gait[:,0] = torch.pi
        self.trot_gait[:,-1] = torch.pi
        print(self.trot_gait)

        self.trot_pattern1 = torch.tensor([1.,0,0,1.],dtype=torch.float, device=self.device,requires_grad=False).view(1,-1)
        self.trot_pattern2 = torch.tensor([0.,1.,1.,0.],dtype=torch.float, device=self.device,requires_grad=False).view(1,-1)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(ROOT_DIR=ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        if len(feet_names) != 4:
            raise RuntimeError(
                f"Expected 4 feet matching asset.foot_name='{self.cfg.asset.foot_name}', "
                f"but found {len(feet_names)}: {feet_names}"
            )

        for s in feet_names:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            if feet_idx < 0:
                raise RuntimeError(f"Could not find foot rigid body '{s}' in asset bodies: {body_names}")
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        print("Creating env...")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.attach_camera(i, env_handle, actor_handle)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            friction_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].friction
            self.friction_coeffs_tensor = friction_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs_tensor = self.restitution_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            restitution_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].restitution
            self.restitution_coeffs_tensor = restitution_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [random.randint(0,self.cfg.domain_rand.lag_timesteps-1) for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False
        else:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [self.cfg.domain_rand.lag_timesteps-1 for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            print(feet_names[i])
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # Track thigh and calf collision geometry for every leg. Any pair of
        # different legs can cross during recovery, not just front calf/rear
        # thigh on the same side.
        recovery_thigh_names = [
            "FL_thigh_Link", "FR_thigh_Link", "RL_thigh_Link", "RR_thigh_Link"
        ]
        recovery_calf_names = [
            "FL_calf_Link", "FR_calf_Link", "RL_calf_Link", "RR_calf_Link"
        ]
        missing_recovery_bodies = [
            name for name in recovery_thigh_names + recovery_calf_names
            if name not in body_names
        ]
        if missing_recovery_bodies:
            raise RuntimeError(
                "Missing bodies required by recovery leg-clearance reward: "
                f"{missing_recovery_bodies}. Asset bodies: {body_names}"
            )
        self.recovery_thigh_indices = torch.tensor([
            self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            ) for name in recovery_thigh_names
        ], dtype=torch.long, device=self.device)
        self.recovery_calf_indices = torch.tensor([
            self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            ) for name in recovery_calf_names
        ], dtype=torch.long, device=self.device)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def reindex(self,tensor):
        #sim2real purpose
        return tensor[:,[3,4,5,0,1,2,9,10,11,6,7,8]]
    
    def reindex_feet(self,tensor):
        return tensor[:,[1,0,3,2]]

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        #self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        #self.cfg.control.action_scale
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)

        actions = self.reindex(actions)
        actions = actions.to(self.device)

        # self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)

        self.global_counter += 1   
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
 
        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.cost_buf,self.reset_buf, self.extras
    
    def compute_observations(self):

        obs_buf =torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                            self.base_ang_vel  * self.obs_scales.ang_vel,
                            self.projected_gravity,
                            self.commands[:, :3] * self.commands_scale,
                            self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            # torch.norm(self.commands[:, :3] * self.commands_scale,dim=-1,keepdim=True)*torch.sin(self.phase),
                            # torch.norm(self.commands[:, :3] * self.commands_scale,dim=-1,keepdim=True)*torch.cos(self.phase),
                            #self.reindex_feet(self.contact_filt.float()-0.5),
                            # self.reindex(self.action_history_buf[:,-1])),dim=-1)
                            self.action_history_buf[:,-1]),dim=-1)

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.zeros(3),
                               torch.ones(3) * noise_scales.ang_vel * noise_level,
                               torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.zeros(3),
                               torch.ones(
                                   12) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   12) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                            #    torch.zeros(4),
                            #    torch.zeros(4),
                               #torch.ones(4) * noise_scales.contact_states * noise_level,
                               #torch.zeros(4),
                               torch.zeros(self.num_actions),
                               ), dim=0)
        
        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * noise_vec.to(self.device)

        priv_latent = torch.cat((
            #self.base_lin_vel * self.obs_scales.lin_vel,
            self.reindex_feet(self.contact_filt.float()-0.5),
            self.randomized_lag_tensor,
            #self.base_ang_vel  * self.obs_scales.ang_vel,
            # self.base_lin_vel * self.obs_scales.lin_vel,
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.restitution_coeffs_tensor,
            self.motor_strength, 
            self.kp_factor,
            self.kd_factor), dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            #priv_latent = torch.cat([priv_latent,self.feet_local_heights],dim=-1)
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)*self.obs_scales.height_measurements
            self.obs_buf = torch.cat([obs_buf, heights, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        # update buffer
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

        if self.cfg.terrain.include_act_obs_pair_buf:
            # add to full observation history and action history to obs
            pure_obs_hist = self.obs_history_buf[:,:,:-self.num_actions].reshape(self.num_envs,-1)
            act_hist = self.action_history_buf.view(self.num_envs,-1)
            self.obs_buf = torch.cat([self.obs_buf,pure_obs_hist,act_hist], dim=-1)
    
    #------------- Callbacks --------------
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        # ``extras`` contains per-step signals consumed by PPO. Keeping the
        # previous reset's episode/time-out data here makes PPO bootstrap every
        # later transition as though it were also a time-out.
        self.extras = {}
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        #self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.compute_cost()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        if self.cfg.env.send_timeouts:
            # Clone the current step's mask; do not leave an aliased/stale mask
            # in extras after time_out_buf is replaced on the following step.
            self.extras["time_outs"] = self.time_out_buf.clone()

        self.update_depth_buffer()
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            
    #------------- Cameras --------------
    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0],config.angle[1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return 
        # not meet the requirement of update
        if self.global_counter % self.cfg.depth.update_interval != 0:
            return 
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)
        
        self.gym.end_access_image_tensors(self.sim)

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._process_phase()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.feet_heights = self._get_feet_heights()
            self.feet_body_frame_height = self._get_feet_local_heights()
            
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        if self.cfg.domain_rand.recovery_push_robots and (self.common_step_counter % self.cfg.domain_rand.recovery_push_interval == 0):
            self._recovery_push_robots()

        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

        self._process_recovery_command_hold()

    def _process_phase(self):
        """update phase value for all actor"""
        self.phase_time = torch.fmod(self.frequency*self.dt + self.phase_time,1.0)
        self.phase = 2*torch.pi*self.phase_time+self.trot_gait
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
     
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare friction randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets,1), device='cpu')
                self.restitution_coeffs = restitution_buckets[bucket_ids]
     
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def _process_rigid_body_props(self, props, env_id):
     
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])

        return props, mass_params
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props
    
    def _low_pass_action_filter(self, actions):
        actons_filtered = self.last_actions * 0.2 + actions * 0.8
        return actons_filtered

    def _get_joint_pd_reference(self):
        """Blend curl -> stand from measured body orientation.

        projected_gravity z is about +1 when the robot is feet-up and -1 when
        upright. Keeping this outside the policy action preserves compatibility
        with the original Panda checkpoint and its standing action space.
        """
        if not getattr(self.cfg.domain_rand, "use_hybrid_recovery_reference", False):
            return self.default_dof_pos

        gravity_z = self.projected_gravity[:, 2]
        blend_start = self.cfg.domain_rand.recovery_reference_blend_start_gz
        blend_full = self.cfg.domain_rand.recovery_reference_blend_full_gz
        curl_blend = torch.clamp(
            (gravity_z - blend_start) / max(blend_full - blend_start, 1e-6),
            0.0,
            1.0,
        ).unsqueeze(1)
        return self.default_dof_pos + curl_blend * (
            self.recovery_curl_dof_pos - self.default_dof_pos
        )
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)

        #pd controller
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction

        # if self.cfg.domain_rand.randomize_lag_timesteps:
        #     self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
        #     joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        # else:
        #     joint_pos_target = actions_scaled + self.default_dof_pos

        joint_reference = self._get_joint_pd_reference()
        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + joint_reference
        else:
            joint_pos_target = actions_scaled + joint_reference

        # joint_pos_target = torch.clamp(joint_pos_target,self.dof_pos-1,self.dof_pos+1)

        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target- self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        torques = torques * self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        self.recovery_success_buf[:] = False
        if getattr(self.cfg.domain_rand, "recovery_only", False):
            self.recovery_elapsed_steps += 1
            base_height = self._get_base_heights()
            upright = -self.projected_gravity[:, 2]
            gravity_z = self.projected_gravity[:, 2]

            # Hysteresis prevents contact vibration around one threshold from
            # being counted as several attempts.  Only explicit feet-up starts
            # use this statistic/reward shaping.
            upside_active = self.recovery_upside_down_active
            escape_threshold = float(getattr(
                self.cfg.domain_rand, "recovery_retry_escape_gravity_z", 0.25
            ))
            return_threshold = float(getattr(
                self.cfg.domain_rand, "recovery_retry_return_gravity_z", 0.65
            ))
            self.recovery_reached_side |= (
                upside_active & (gravity_z < escape_threshold)
            )
            retry_event = (
                upside_active
                & self.recovery_reached_side
                & (gravity_z > return_threshold)
            )
            self.recovery_retry_count += retry_event.long()
            self.recovery_reached_side[retry_event] = False

            contact_count = torch.sum(self.contact_filt.float(), dim=1)
            lin_speed = torch.norm(self.base_lin_vel, dim=1)
            ang_speed = torch.norm(self.base_ang_vel, dim=1)
            pose_error = torch.abs(self.dof_pos - self.default_dof_pos)
            pose_rmse = torch.sqrt(
                torch.mean(torch.square(pose_error), dim=1) + 1e-8
            )
            max_joint_error = torch.max(pose_error, dim=1).values
            pose_ok = (
                (pose_rmse < self.cfg.domain_rand.recovery_success_pose_rmse)
                & (max_joint_error
                   < self.cfg.domain_rand.recovery_success_max_joint_error)
            )
            stable = (
                (base_height > self.cfg.domain_rand.recovery_success_height_ratio
                 * self.cfg.rewards.base_height_target)
                & (upright > self.cfg.domain_rand.recovery_success_upright)
                & (contact_count >= self.cfg.domain_rand.recovery_success_contacts)
                & (lin_speed < self.cfg.domain_rand.recovery_success_max_lin_vel)
                & (ang_speed < self.cfg.domain_rand.recovery_success_max_ang_vel)
                & pose_ok
            )
            self.recovery_stable_steps = torch.where(
                stable,
                self.recovery_stable_steps + 1,
                torch.zeros_like(self.recovery_stable_steps),
            )
            success = self.recovery_stable_steps >= self.cfg.domain_rand.recovery_success_hold_steps
            new_success = success & (~self.recovery_episode_success)
            # This is intentionally a one-step event. It makes the terminal
            # success reward truly one-off instead of paying on every remaining
            # stable step of the episode.
            self.recovery_success_buf[new_success] = True
            self.recovery_first_success_step[new_success] = (
                self.recovery_elapsed_steps[new_success]
            )
            self.recovery_episode_success |= success
            if getattr(
                self.cfg.domain_rand, "recovery_terminate_on_success", True
            ):
                self.reset_buf |= success

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_cost(self):
        self.cost_buf[:] = 0
        for i in range(len(self.cost_functions)):
            name = self.cost_names[i]
            cost = self.cost_functions[i]() * self.dt #self.cost_scales[name]
            self.cost_buf[:,i] += cost
            self.cost_episode_sums[name] += cost
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        ended_episode_lengths = self.episode_length_buf[env_ids].clone()
        ended_success_mask = self.recovery_episode_success[env_ids].clone()
        ended_upside_mask = self.recovery_upside_down_active[env_ids].clone()
        ended_exact_upside_mask = self.recovery_exact_upside_active[
            env_ids
        ].clone()
        ended_left_side_mask = self.recovery_left_side_active[env_ids].clone()
        ended_right_side_mask = self.recovery_right_side_active[env_ids].clone()
        ended_retry_count = self.recovery_retry_count[env_ids].clone()
        ended_first_success_step = self.recovery_first_success_step[
            env_ids
        ].clone()
        ended_min_leg_clearance = self.recovery_episode_min_leg_clearance[
            env_ids
        ].clone()
        self._update_recovery_success_curriculum(
            ended_success_mask, ended_episode_lengths
        )
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self._update_command_curriculum(env_ids)

        # Clear recovery state before root sampling. Recovery-only root sampling
        # will mark the environments that actually start feet-up again.
        self.recovery_command_timer[env_ids] = 0
        self.recovery_upside_down_active[env_ids] = False
        self.recovery_exact_upside_active[env_ids] = False
        self.recovery_left_side_active[env_ids] = False
        self.recovery_right_side_active[env_ids] = False
        self.recovery_last_gravity_z[env_ids] = 1.
        self.recovery_stable_steps[env_ids] = 0
        self.recovery_episode_success[env_ids] = False
        self.recovery_retry_count[env_ids] = 0
        self.recovery_reached_side[env_ids] = False
        self.recovery_first_success_step[env_ids] = -1
        self.recovery_elapsed_steps[env_ids] = 0
        self.recovery_episode_min_leg_clearance[env_ids] = float("inf")

        # Reset the root first so recovery_upside_down_active is available to
        # _reset_dofs. Feet-up episodes can then start in FR-Net's curled pose,
        # while all other episodes retain Panda's standing action reference.
        self._reset_root_states(env_ids)
        self._reset_dofs(env_ids)
        self._resample_commands(env_ids)

        if getattr(self.cfg.domain_rand, "recovery_only", False):
            self.recovery_command_timer[env_ids] = self.max_episode_length + 1
            self.recovery_commands[env_ids] = 0.
            self.commands[env_ids] = 0.
        elif self.cfg.domain_rand.randomize_init_state and self.cfg.domain_rand.recovery_push_robots:
            self.recovery_commands[env_ids] = self.commands[env_ids]
            self.recovery_command_timer[env_ids] = int(self.cfg.domain_rand.recovery_command_hold_steps)
            self.commands[env_ids] = 0.
        else:
            self.recovery_command_timer[env_ids] = 0

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        for key in self.cost_episode_sums.keys():
            self.extras["episode"]['cost_'+ key] = torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
            self.cost_episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if getattr(self.cfg.domain_rand, "recovery_only", False):
            success_mask = ended_success_mask
            self.extras["episode"]["recovery_success_rate"] = success_mask.float().mean()
            curriculum_progress, _, curriculum_max_tilt = (
                self._get_recovery_curriculum_values()
            )
            hard_start = float(getattr(
                self.cfg.domain_rand, "recovery_curriculum_hard_start", 0.20
            ))
            if self.cfg.domain_rand.recovery_curriculum:
                raw_hard_progress = np.clip(
                    (curriculum_progress - hard_start)
                    / max(1.0 - hard_start, 1e-6), 0.0, 1.0
                )
                initial_hard_ratio = float(getattr(
                    self.cfg.domain_rand,
                    "recovery_curriculum_initial_hard_ratio", 0.0
                ))
                hard_progress = initial_hard_ratio + (
                    1.0 - initial_hard_ratio
                ) * raw_hard_progress
            else:
                hard_progress = 1.0
            self.extras["episode"]["recovery_curriculum_progress"] = torch.tensor(
                curriculum_progress, device=self.device
            )
            self.extras["episode"]["recovery_curriculum_stage"] = torch.tensor(
                self.recovery_curriculum_stage, device=self.device
            )
            self.extras["episode"]["recovery_curriculum_success_ema"] = torch.tensor(
                self.recovery_curriculum_success_ema, device=self.device
            )
            self.extras["episode"]["recovery_curriculum_max_tilt_rad"] = torch.tensor(
                curriculum_max_tilt, device=self.device
            )
            self.extras["episode"]["recovery_target_upside_ratio"] = torch.tensor(
                self.cfg.domain_rand.recovery_upside_down_ratio * hard_progress,
                device=self.device
            )
            self.extras["episode"]["recovery_initial_upside_ratio"] = ended_upside_mask.float().mean()
            self.extras["episode"]["recovery_initial_exact_upside_ratio"] = (
                ended_exact_upside_mask.float().mean()
            )
            if torch.any(ended_upside_mask):
                self.extras["episode"]["recovery_upside_success_rate"] = (
                    success_mask[ended_upside_mask].float().mean()
                )
                self.extras["episode"]["recovery_upside_first_attempt_success_rate"] = (
                    (
                        success_mask[ended_upside_mask]
                        & (ended_retry_count[ended_upside_mask] == 0)
                    ).float().mean()
                )
                self.extras["episode"]["recovery_upside_retry_count"] = (
                    ended_retry_count[ended_upside_mask].float().mean()
                )
            else:
                self.extras["episode"]["recovery_upside_success_rate"] = torch.tensor(
                    0., device=self.device
                )
                self.extras["episode"]["recovery_upside_first_attempt_success_rate"] = torch.tensor(
                    0., device=self.device
                )
                self.extras["episode"]["recovery_upside_retry_count"] = torch.tensor(
                    0., device=self.device
                )
            if torch.any(ended_exact_upside_mask):
                self.extras["episode"]["recovery_exact_upside_success_rate"] = (
                    success_mask[ended_exact_upside_mask].float().mean()
                )
                self.extras["episode"]["recovery_exact_upside_first_attempt_success_rate"] = (
                    (
                        success_mask[ended_exact_upside_mask]
                        & (ended_retry_count[ended_exact_upside_mask] == 0)
                    ).float().mean()
                )
            else:
                self.extras["episode"]["recovery_exact_upside_success_rate"] = torch.tensor(
                    0., device=self.device
                )
                self.extras["episode"]["recovery_exact_upside_first_attempt_success_rate"] = torch.tensor(
                    0., device=self.device
                )
            if torch.any(ended_left_side_mask):
                self.extras["episode"]["recovery_left_side_success_rate"] = (
                    success_mask[ended_left_side_mask].float().mean()
                )
            else:
                self.extras["episode"]["recovery_left_side_success_rate"] = torch.tensor(
                    0., device=self.device
                )
            if torch.any(ended_right_side_mask):
                self.extras["episode"]["recovery_right_side_success_rate"] = (
                    success_mask[ended_right_side_mask].float().mean()
                )
            else:
                self.extras["episode"]["recovery_right_side_success_rate"] = torch.tensor(
                    0., device=self.device
                )
            if torch.any(success_mask):
                self.extras["episode"]["recovery_time_s"] = (
                    ended_first_success_step[success_mask].float().mean()
                    * self.dt
                )
            else:
                self.extras["episode"]["recovery_time_s"] = torch.tensor(
                    0., device=self.device
                )
            finite_clearance = torch.where(
                torch.isfinite(ended_min_leg_clearance),
                ended_min_leg_clearance,
                torch.zeros_like(ended_min_leg_clearance)
            )
            self.extras["episode"]["recovery_leg_min_clearance_m"] = (
                torch.amin(finite_clearance, dim=1).mean()
            )
            episode_min_clearance = torch.amin(finite_clearance, dim=1)
            if torch.any(success_mask):
                self.extras["episode"]["recovery_success_min_clearance_m"] = (
                    episode_min_clearance[success_mask].mean()
                )
            else:
                self.extras["episode"]["recovery_success_min_clearance_m"] = (
                    torch.tensor(0., device=self.device)
                )
            failed_mask = ~success_mask
            if torch.any(failed_mask):
                self.extras["episode"]["recovery_failed_min_clearance_m"] = (
                    episode_min_clearance[failed_mask].mean()
                )
                self.extras["episode"]["recovery_failed_contact_rate"] = (
                    episode_min_clearance[failed_mask] <= 0.01
                ).float().mean()
            else:
                self.extras["episode"]["recovery_failed_min_clearance_m"] = (
                    torch.tensor(0., device=self.device)
                )
                self.extras["episode"]["recovery_failed_contact_rate"] = (
                    torch.tensor(0., device=self.device)
                )
            pair_names = ("FL_RL", "FL_RR", "FR_RL", "FR_RR")
            for pair_index, pair_name in enumerate(pair_names):
                self.extras["episode"][
                    f"recovery_{pair_name}_surface_clearance_m"
                ] = finite_clearance[:, pair_index].mean()
            self.recovery_success_buf[env_ids] = False
        # for i in range(len(self.lag_buffer)):
        #     self.lag_buffer[i][env_ids, :] = 0
        self.lag_buffer[env_ids,:,:] = 0
        self.phase[env_ids,:] = 0
        self.phase_time[env_ids,:] = 0
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs,_,_, _, _,_= self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if (self.cfg.domain_rand.randomize_init_state
                and getattr(self.cfg.domain_rand, "recovery_only", False)):
            self._reset_recovery_root_states(env_ids)
        elif self.cfg.domain_rand.randomize_init_state:
            # base velocities
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
            progress, _, max_tilt = self._get_recovery_curriculum_values()
            full_random_ratio = 0.1 * progress
            full_random_envs = torch.rand(len(env_ids), device=self.device) < full_random_ratio
            limited_env_ids = env_ids[~full_random_envs]
            full_env_ids = env_ids[full_random_envs]

            if len(limited_env_ids) > 0:
                roll = torch_rand_float(-max_tilt, max_tilt, (len(limited_env_ids), 1), device=self.device).squeeze(1)
                pitch = torch_rand_float(-max_tilt, max_tilt, (len(limited_env_ids), 1), device=self.device).squeeze(1)
                yaw = torch_rand_float(-np.pi, np.pi, (len(limited_env_ids), 1), device=self.device).squeeze(1)
                self.root_states[limited_env_ids, 3:7] = quat_from_euler_xyz_local(roll, pitch, yaw)

            if len(full_env_ids) > 0:
                self.root_states[full_env_ids, 3:7] = random_quat(torch_rand_float(0, 1, (len(full_env_ids), 4), device=self.device))

            self.root_states[env_ids, 2:3] += torch_rand_float(0, 0.2, (len(env_ids), 1), device=self.device) 
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_recovery_root_states(self, env_ids):
        """Sample static feet-up poses plus arbitrary fallen recovery poses."""
        count = len(env_ids)
        if count == 0:
            return

        final_upside_ratio = float(
            self.cfg.domain_rand.recovery_upside_down_ratio
        )
        final_left_ratio = float(getattr(
            self.cfg.domain_rand, "recovery_left_side_ratio", 0.0
        ))
        final_right_ratio = float(getattr(
            self.cfg.domain_rand, "recovery_right_side_ratio", 0.0
        ))
        if (final_upside_ratio + final_left_ratio + final_right_ratio
                > 1.0 + 1e-6):
            raise ValueError(
                "Recovery upside/left/right reset ratios must sum to <= 1, "
                f"got {final_upside_ratio + final_left_ratio + final_right_ratio:.3f}"
            )
        progress, _, max_tilt = self._get_recovery_curriculum_values()
        hard_start = float(getattr(
            self.cfg.domain_rand, "recovery_curriculum_hard_start", 0.20
        ))
        if self.cfg.domain_rand.recovery_curriculum:
            raw_hard_progress = np.clip(
                (progress - hard_start) / max(1.0 - hard_start, 1e-6),
                0.0, 1.0
            )
            initial_hard_ratio = float(getattr(
                self.cfg.domain_rand,
                "recovery_curriculum_initial_hard_ratio", 0.0
            ))
            hard_progress = initial_hard_ratio + (
                1.0 - initial_hard_ratio
            ) * raw_hard_progress
        else:
            hard_progress = 1.0
        upside_ratio = final_upside_ratio * hard_progress
        left_ratio = final_left_ratio * hard_progress
        right_ratio = final_right_ratio * hard_progress
        reset_sample = torch.rand(count, device=self.device)
        upside_mask = reset_sample < upside_ratio
        left_side_mask = (
            (reset_sample >= upside_ratio)
            & (reset_sample < upside_ratio + left_ratio)
        )
        right_side_mask = (
            (reset_sample >= upside_ratio + left_ratio)
            & (reset_sample < upside_ratio + left_ratio + right_ratio)
        )
        generic_mask = ~(upside_mask | left_side_mask | right_side_mask)
        upside_ids = env_ids[upside_mask]
        left_side_ids = env_ids[left_side_mask]
        right_side_ids = env_ids[right_side_mask]
        generic_ids = env_ids[generic_mask]

        # Arbitrary fallen states retain a small physical velocity. Exact
        # feet-up states are static so the policy must create the escape motion.
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.2, 0.2, (count, 6), device=self.device
        )
        if len(generic_ids) > 0:
            roll = torch_rand_float(
                -max_tilt, max_tilt, (len(generic_ids), 1),
                device=self.device
            ).squeeze(1)
            pitch = torch_rand_float(
                -max_tilt, max_tilt, (len(generic_ids), 1),
                device=self.device
            ).squeeze(1)
            yaw = torch_rand_float(
                -np.pi, np.pi, (len(generic_ids), 1),
                device=self.device
            ).squeeze(1)
            self.root_states[generic_ids, 3:7] = quat_from_euler_xyz_local(
                roll, pitch, yaw
            )
            final_height_range = self.cfg.domain_rand.recovery_randomize_height_range
            initial_height_range = getattr(
                self.cfg.domain_rand,
                "recovery_curriculum_initial_height_range",
                final_height_range
            )
            height_range = [
                initial_height_range[0] + progress * (
                    final_height_range[0] - initial_height_range[0]
                ),
                initial_height_range[1] + progress * (
                    final_height_range[1] - initial_height_range[1]
                ),
            ]
            self.root_states[generic_ids, 2:3] = (
                self.env_origins[generic_ids, 2:3]
                + torch_rand_float(
                    height_range[0], height_range[1],
                    (len(generic_ids), 1), device=self.device
                )
            )

        # Explicit, static side-lying starts prevent aggregate random
        # orientations from hiding a strong left/right recovery asymmetry.
        side_ids = torch.cat((left_side_ids, right_side_ids))
        if len(side_ids) > 0:
            left_count = len(left_side_ids)
            right_count = len(right_side_ids)
            roll = torch.cat((
                torch.full(
                    (left_count,), -0.5 * np.pi, device=self.device
                ),
                torch.full(
                    (right_count,), 0.5 * np.pi, device=self.device
                ),
            ))
            roll_noise = float(getattr(
                self.cfg.domain_rand, "recovery_side_roll_noise", 0.15
            ))
            pitch_noise = float(getattr(
                self.cfg.domain_rand, "recovery_side_pitch_noise", 0.10
            ))
            roll += torch_rand_float(
                -roll_noise, roll_noise, (len(side_ids), 1), device=self.device
            ).squeeze(1)
            pitch = torch_rand_float(
                -pitch_noise, pitch_noise,
                (len(side_ids), 1), device=self.device
            ).squeeze(1)
            yaw = torch_rand_float(
                -np.pi, np.pi, (len(side_ids), 1), device=self.device
            ).squeeze(1)
            self.root_states[side_ids, 3:7] = quat_from_euler_xyz_local(
                roll, pitch, yaw
            )
            side_height_range = getattr(
                self.cfg.domain_rand,
                "recovery_side_height_range", [0.20, 0.32]
            )
            self.root_states[side_ids, 2:3] = (
                self.env_origins[side_ids, 2:3]
                + torch_rand_float(
                    side_height_range[0], side_height_range[1],
                    (len(side_ids), 1), device=self.device
                )
            )
            # These cases must be escaped by the policy itself, not by an
            # initial angular impulse that will not exist on hardware.
            self.root_states[side_ids, 7:13] = 0.
            self.recovery_left_side_active[left_side_ids] = True
            self.recovery_right_side_active[right_side_ids] = True

        if len(upside_ids) > 0:
            upside_count = len(upside_ids)
            angle_noise = self.cfg.domain_rand.recovery_upside_down_angle_noise
            tilt_noise = self.cfg.domain_rand.recovery_upside_down_tilt_noise
            exact_ratio = float(getattr(
                self.cfg.domain_rand, "recovery_exact_upside_ratio", 0.0
            ))
            exact_mask = (
                torch.rand(upside_count, device=self.device) < exact_ratio
            )
            self.recovery_exact_upside_active[upside_ids[exact_mask]] = True
            flip_roll = torch.rand(upside_count, device=self.device) < 0.5
            flip_sign = torch.where(
                torch.rand(upside_count, device=self.device) < 0.5,
                -torch.ones(upside_count, device=self.device),
                torch.ones(upside_count, device=self.device),
            )
            flip_angle = flip_sign * (
                np.pi - torch_rand_float(
                    0., angle_noise, (upside_count, 1), device=self.device
                ).squeeze(1)
            )
            small_roll = torch_rand_float(
                -tilt_noise, tilt_noise, (upside_count, 1), device=self.device
            ).squeeze(1)
            small_pitch = torch_rand_float(
                -tilt_noise, tilt_noise, (upside_count, 1), device=self.device
            ).squeeze(1)
            roll = torch.where(flip_roll, flip_angle, small_roll)
            pitch = torch.where(flip_roll, small_pitch, flip_angle)
            # Real falls often settle at the perfectly symmetric, static 180
            # degree pose. Continuous noise sampling has zero probability of
            # producing that exact local minimum, so replay it explicitly.
            exact_flip_angle = flip_sign * torch.full_like(flip_angle, np.pi)
            roll = torch.where(
                exact_mask,
                torch.where(flip_roll, exact_flip_angle, torch.zeros_like(roll)),
                roll,
            )
            pitch = torch.where(
                exact_mask,
                torch.where(flip_roll, torch.zeros_like(pitch), exact_flip_angle),
                pitch,
            )
            yaw = torch_rand_float(
                -np.pi, np.pi, (upside_count, 1), device=self.device
            ).squeeze(1)
            self.root_states[upside_ids, 3:7] = quat_from_euler_xyz_local(
                roll, pitch, yaw
            )
            upside_height_range = self.cfg.domain_rand.recovery_upside_down_height_range
            self.root_states[upside_ids, 2:3] = (
                self.env_origins[upside_ids, 2:3]
                + torch_rand_float(
                    upside_height_range[0], upside_height_range[1],
                    (upside_count, 1), device=self.device
                )
            )
            self.root_states[upside_ids, 7:13] = 0.

        initial_gravity_z = quat_rotate_inverse(
            self.root_states[env_ids, 3:7], self.gravity_vec[env_ids]
        )[:, 2]
        self.recovery_upside_down_active[env_ids] = initial_gravity_z > 0.5
        self.recovery_last_gravity_z[env_ids] = initial_gravity_z
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        if getattr(self.cfg.domain_rand, "recovery_only", False):
            # Hip defaults are zero, so multiplicative reset randomization left
            # every fallen training pose with perfectly symmetric hips. Add an
            # independent offset to cover the splayed/asymmetric hardware poses
            # without changing the deployment-compatible standing reference.
            hip_noise = float(getattr(
                self.cfg.domain_rand, "recovery_reset_hip_noise", 0.18
            ))
            if len(self.recovery_hip_dof_indices) > 0 and hip_noise > 0.:
                hip_offsets = torch_rand_float(
                    -hip_noise, hip_noise,
                    (len(env_ids), len(self.recovery_hip_dof_indices)),
                    device=self.device,
                )
                self.dof_pos[
                    env_ids.unsqueeze(1), self.recovery_hip_dof_indices
                ] += hip_offsets
                self.dof_pos[env_ids] = torch.max(
                    torch.min(self.dof_pos[env_ids], self.dof_pos_limits[:, 1]),
                    self.dof_pos_limits[:, 0],
                )
        # Match the environment used to train the model_3600 baseline.  With
        # the deployment-compatible standing reference, recovery starts retain
        # the ordinary randomized standing-pose DOFs instead of being replaced
        # by the later experimental curled-pose distribution.
        if (getattr(self.cfg.domain_rand, "recovery_only", False)
                and getattr(self.cfg.domain_rand, "use_hybrid_recovery_reference", False)):
            upside_ids = env_ids[self.recovery_upside_down_active[env_ids]]
            if len(upside_ids) > 0:
                num_upside = len(upside_ids)
                joint_noise = float(getattr(
                    self.cfg.domain_rand, "recovery_upside_joint_noise", 0.20
                ))
                hip_noise = float(getattr(
                    self.cfg.domain_rand, "recovery_upside_hip_noise", 0.12
                ))
                noise_scale = torch.full(
                    (1, self.num_dof), joint_noise,
                    dtype=self.dof_pos.dtype, device=self.device
                )
                for i, name in enumerate(self.dof_names):
                    if "hip_joint" in name:
                        noise_scale[0, i] = hip_noise

                curl_pose = self.recovery_curl_dof_pos.repeat(num_upside, 1)
                curl_pose += torch_rand_float(
                    -1.0, 1.0, (num_upside, self.num_dof), device=self.device
                ) * noise_scale

                # A minority of resets spans the curl-to-stand interval. This
                # adds asymmetric and partially open poses without replacing
                # the proven compact start distribution.
                broad_ratio = float(getattr(
                    self.cfg.domain_rand,
                    "recovery_upside_random_joint_ratio", 0.25
                ))
                broad_mask = (
                    torch.rand(num_upside, 1, device=self.device) < broad_ratio
                )
                broad_alpha = torch.rand(
                    num_upside, self.num_dof, device=self.device
                )
                broad_pose = self.recovery_stand_dof_pos + broad_alpha * (
                    self.recovery_curl_dof_pos - self.recovery_stand_dof_pos
                )
                broad_pose += torch_rand_float(
                    -0.5, 0.5, (num_upside, self.num_dof), device=self.device
                ) * noise_scale
                upside_pose = torch.where(broad_mask, broad_pose, curl_pose)
                upside_pose = torch.max(
                    torch.min(upside_pose, self.dof_pos_limits[:, 1]),
                    self.dof_pos_limits[:, 0]
                )
                self.dof_pos[upside_ids] = upside_pose

            # Side-lying and other clearly fallen starts also need varied hip
            # and fold configurations. Previously hip remained exactly zero
            # for these states, so the hardware's splayed/locked poses were
            # outside the training reset distribution.
            initial_gravity_z = quat_rotate_inverse(
                self.root_states[env_ids, 3:7], self.gravity_vec[env_ids]
            )[:, 2]
            fallen_mask = (
                (initial_gravity_z > -0.7)
                & (~self.recovery_upside_down_active[env_ids])
            )
            fallen_ids = env_ids[fallen_mask]
            if len(fallen_ids) > 0:
                num_fallen = len(fallen_ids)
                side_joint_noise = float(getattr(
                    self.cfg.domain_rand, "recovery_side_joint_noise", 0.25
                ))
                side_hip_noise = float(getattr(
                    self.cfg.domain_rand, "recovery_side_hip_noise", 0.20
                ))
                side_noise_scale = torch.full(
                    (1, self.num_dof), side_joint_noise,
                    dtype=self.dof_pos.dtype, device=self.device
                )
                for i, name in enumerate(self.dof_names):
                    if "hip_joint" in name:
                        side_noise_scale[0, i] = side_hip_noise
                fold = torch.rand(
                    num_fallen, self.num_dof, device=self.device
                )
                fallen_pose = self.recovery_stand_dof_pos + fold * (
                    self.recovery_curl_dof_pos - self.recovery_stand_dof_pos
                )
                fallen_pose += torch_rand_float(
                    -1.0, 1.0, (num_fallen, self.num_dof), device=self.device
                ) * side_noise_scale
                fallen_pose = torch.max(
                    torch.min(fallen_pose, self.dof_pos_limits[:, 1]),
                    self.dof_pos_limits[:, 0]
                )
                self.dof_pos[fallen_ids] = fallen_pose
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        
    def _prepare_cost_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.cost_scales.keys()):
            scale = self.cost_scales[key]
            if scale==0:
                self.cost_scales.pop(key) 
            # else:
            #     self.cost_scales[key] *= self.dt

        self.cost_functions = []
        self.cost_names = []
        self.cost_k_values = []
        self.cost_d_values_tensor = []

        for name,scale in self.cost_scales.items():
            self.cost_names.append(name)
            name = '_cost_' + name
            print('cost name:',name)
            print('cost k value:',scale)
            self.cost_functions.append(getattr(self, name))
            self.cost_k_values.append(float(scale))

        for name,value in self.cost_d_values.items():
            print('cost name:',name)
            print('cost d value:',value)
            self.cost_d_values_tensor.append(float(value))

        self.cost_k_values = torch.FloatTensor(self.cost_k_values).view(1,-1).to(self.device)
        self.cost_d_values_tensor = torch.FloatTensor(self.cost_d_values_tensor).view(1,1,-1).to(self.device)

        self.cost_episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                  for name in self.cost_scales.keys()}

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
    
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.cost_scales = class_to_dict(self.cfg.costs.scales)
        self.cost_d_values = class_to_dict(self.cfg.costs.d_values)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        
        # global counter 是否该类似这个
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.recovery_push_interval = np.ceil(self.cfg.domain_rand.recovery_push_interval_s / self.dt)
        self.cfg.domain_rand.recovery_command_hold_steps = np.ceil(self.cfg.domain_rand.recovery_command_hold_s / self.dt)
        if getattr(self.cfg.domain_rand, "recovery_only", False):
            self.cfg.domain_rand.recovery_success_hold_steps = int(np.ceil(
                self.cfg.domain_rand.recovery_success_hold_s / self.dt
            ))

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        # draw depth image with window created by cv2
        if self.cfg.depth.use_camera:
            window_name = "Depth Image"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cup().numpy() + 0.5)
            cv2.waitKey(1) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height
    
    def _get_feet_local_heights(self, env_ids=None):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)

        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])

        return footpos_in_body_frame[:,:,2].view(self.num_envs,-1)
    
    #------------ curriculum ----------------
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _get_recovery_curriculum_values(self):
        progress = 1.
        randomize_ratio = self.cfg.domain_rand.recovery_randomize_orientation_ratio
        max_tilt = self.cfg.domain_rand.recovery_curriculum_final_tilt

        if self.cfg.domain_rand.recovery_curriculum:
            if getattr(
                self.cfg.domain_rand,
                "recovery_curriculum_success_gated", False
            ):
                stages = self.cfg.domain_rand.recovery_curriculum_stage_progress
                stage = min(self.recovery_curriculum_stage, len(stages) - 1)
                progress = float(stages[stage])
            else:
                progress = min(
                    float(self.global_counter)
                    / float(self.cfg.domain_rand.recovery_curriculum_steps), 1.
                )
            start_ratio = self.cfg.domain_rand.recovery_curriculum_initial_orientation_ratio
            randomize_ratio = start_ratio + progress * (randomize_ratio - start_ratio)
            start_tilt = self.cfg.domain_rand.recovery_curriculum_initial_tilt
            final_tilt = self.cfg.domain_rand.recovery_curriculum_final_tilt
            max_tilt = start_tilt + progress * (final_tilt - start_tilt)

        return progress, randomize_ratio, max_tilt

    def _update_recovery_success_curriculum(
            self, ended_success_mask, ended_episode_lengths):
        """Advance recovery difficulty only after the current stage is mastered."""
        if not (
            getattr(self.cfg.domain_rand, "recovery_only", False)
            and self.cfg.domain_rand.recovery_curriculum
            and getattr(
                self.cfg.domain_rand,
                "recovery_curriculum_success_gated", False
            )
        ):
            return

        stages = self.cfg.domain_rand.recovery_curriculum_stage_progress
        if self.recovery_curriculum_stage >= len(stages) - 1:
            return

        # init_at_random_ep_len deliberately creates shortened first episodes
        # to stagger logging. Exclude those artificial endings from mastery;
        # genuine early successful terminations remain valid.
        full_episode = ended_episode_lengths >= self.max_episode_length - 1
        valid = full_episode | ended_success_mask
        if not torch.any(valid):
            return

        batch_success = ended_success_mask[valid].float().mean().item()
        alpha = float(
            self.cfg.domain_rand.recovery_curriculum_success_ema_alpha
        )
        self.recovery_curriculum_success_ema = (
            (1.0 - alpha) * self.recovery_curriculum_success_ema
            + alpha * batch_success
        )
        stage_age = (
            self.common_step_counter - self.recovery_curriculum_stage_start_step
        )
        if (
            stage_age >= self.cfg.domain_rand.recovery_curriculum_min_stage_steps
            and self.recovery_curriculum_success_ema
                >= self.cfg.domain_rand.recovery_curriculum_success_threshold
        ):
            self.recovery_curriculum_stage += 1
            self.recovery_curriculum_success_ema = 0.0
            self.recovery_curriculum_stage_start_step = self.common_step_counter

    def _recovery_push_robots(self):
        """Push robots into off-nominal states during an episode, then hold commands at zero."""
        progress, randomize_ratio, max_tilt = self._get_recovery_curriculum_values()
        recovery_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.recovery_upside_down_active[:] = False
        self.recovery_last_gravity_z[:] = 1.

        max_lin_vel = self.cfg.domain_rand.recovery_push_lin_vel_xy * (0.5 + 0.5 * progress)
        max_ang_vel_xy = self.cfg.domain_rand.recovery_push_ang_vel_xy * (0.4 + 0.6 * progress)
        max_ang_vel_z = self.cfg.domain_rand.recovery_push_ang_vel_z

        self.root_states[:, 7:9] = torch_rand_float(-max_lin_vel, max_lin_vel, (self.num_envs, 2), device=self.device)
        self.root_states[:, 10:12] = torch_rand_float(-max_ang_vel_xy, max_ang_vel_xy, (self.num_envs, 2), device=self.device)
        self.root_states[:, 12:13] = torch_rand_float(-max_ang_vel_z, max_ang_vel_z, (self.num_envs, 1), device=self.device)

        if randomize_ratio > 0.:
            randomize_envs = torch.rand(self.num_envs, device=self.device) < randomize_ratio
            if torch.any(randomize_envs):
                randomize_ids = randomize_envs.nonzero(as_tuple=False).flatten()
                recovery_ids = randomize_ids
                num_randomize = len(randomize_ids)
                height_range = self.cfg.domain_rand.recovery_randomize_height_range
                upside_down_ratio = self.cfg.domain_rand.recovery_upside_down_ratio
                upside_down_mask = torch.rand(num_randomize, device=self.device) < upside_down_ratio
                upside_down_ids = randomize_ids[upside_down_mask]
                generic_ids = randomize_ids[~upside_down_mask]

                if len(generic_ids) > 0:
                    roll = torch_rand_float(-max_tilt, max_tilt, (len(generic_ids), 1), device=self.device).squeeze(1)
                    pitch = torch_rand_float(-max_tilt, max_tilt, (len(generic_ids), 1), device=self.device).squeeze(1)
                    yaw = torch_rand_float(-np.pi, np.pi, (len(generic_ids), 1), device=self.device).squeeze(1)
                    self.root_states[generic_ids, 3:7] = quat_from_euler_xyz_local(roll, pitch, yaw)

                if len(upside_down_ids) > 0:
                    count = len(upside_down_ids)
                    self.recovery_upside_down_active[upside_down_ids] = True
                    angle_noise = self.cfg.domain_rand.recovery_upside_down_angle_noise
                    tilt_noise = self.cfg.domain_rand.recovery_upside_down_tilt_noise
                    flip_roll = torch.rand(count, device=self.device) < 0.5
                    flip_sign = torch.where(
                        torch.rand(count, device=self.device) < 0.5,
                        -torch.ones(count, device=self.device),
                        torch.ones(count, device=self.device),
                    )
                    flip_angle = flip_sign * (
                        np.pi - torch_rand_float(0., angle_noise, (count, 1), device=self.device).squeeze(1)
                    )
                    small_roll = torch_rand_float(-tilt_noise, tilt_noise, (count, 1), device=self.device).squeeze(1)
                    small_pitch = torch_rand_float(-tilt_noise, tilt_noise, (count, 1), device=self.device).squeeze(1)
                    roll = torch.where(flip_roll, flip_angle, small_roll)
                    pitch = torch.where(flip_roll, small_pitch, flip_angle)
                    yaw = torch_rand_float(-np.pi, np.pi, (count, 1), device=self.device).squeeze(1)
                    self.root_states[upside_down_ids, 3:7] = quat_from_euler_xyz_local(roll, pitch, yaw)

                self.root_states[randomize_ids, 2:3] = (
                    self.env_origins[randomize_ids, 2:3]
                    + torch_rand_float(height_range[0], height_range[1], (num_randomize, 1), device=self.device)
                )

                # The explicit feet-up subset is a static recovery problem:
                # place it close to the ground and remove launch velocities.
                # Other randomized falls keep the original dynamic disturbance.
                if len(upside_down_ids) > 0:
                    upside_height_range = getattr(
                        self.cfg.domain_rand, "recovery_upside_down_height_range", height_range
                    )
                    self.root_states[upside_down_ids, 2:3] = (
                        self.env_origins[upside_down_ids, 2:3]
                        + torch_rand_float(
                            upside_height_range[0], upside_height_range[1],
                            (len(upside_down_ids), 1), device=self.device
                        )
                    )
                    self.root_states[upside_down_ids, 7:13] = 0.

        # Only pose-randomized robots need a dedicated zero-command recovery
        # window. Other robots keep tracking velocity after the impulse.
        self.recovery_commands[recovery_ids] = self.commands[recovery_ids]
        self.recovery_command_timer[recovery_ids] = int(self.cfg.domain_rand.recovery_command_hold_steps)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _process_recovery_command_hold(self):
        """Keep commands zero for a short recovery window after a push."""
        if getattr(self.cfg.domain_rand, "recovery_only", False):
            self.commands[:] = 0.
            self.recovery_command_timer[:] = self.max_episode_length + 1
            return

        active = self.recovery_command_timer > 0
        if torch.any(active):
            self.commands[active, :] = 0.
            self.recovery_command_timer[active] -= 1
            finished = active & (self.recovery_command_timer == 0)
            if torch.any(finished):
                self.commands[finished] = self.recovery_commands[finished]
                self.recovery_upside_down_active[finished] = False
                self.recovery_last_gravity_z[finished] = 1.

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # If a command is resampled during a recovery hold, remember the new
        # target and expose it only after the robot's stand-up window ends.
        active = self.recovery_command_timer[env_ids] > 0
        if torch.any(active):
            active_ids = env_ids[active]
            self.recovery_commands[active_ids] = self.commands[active_ids]
            self.commands[active_ids] = 0.
    
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            # self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1, -self.cfg.commands.max_curriculum, 0.)
            # self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0., self.cfg.commands.max_curriculum)
            # self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.1, -self.cfg.commands.max_curriculum, 0.)
            # self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.1, 0., self.cfg.commands.max_curriculum)

            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1, -self.cfg.commands.max_backward_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0., self.cfg.commands.max_forward_curriculum)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.1, -self.cfg.commands.max_lat_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.1, 0., self.cfg.commands.max_lat_curriculum)


    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height

    #------------ reward functions----------------
    def _recovery_reward_active(self):
        """One during the zero-command recovery hold, otherwise zero."""
        if getattr(self.cfg.domain_rand, "recovery_only", False):
            return torch.ones(self.num_envs, device=self.device)
        # The original Go2 configuration learns recovery without the later
        # dedicated recovery-push/command-hold mechanism. Its upward/contact
        # rewards are global and must therefore remain active. Without this
        # compatibility branch, copying go2_constraint_him would silently make
        # both rewards identically zero.
        if not getattr(self.cfg.domain_rand, "recovery_push_robots", False):
            return torch.ones(self.num_envs, device=self.device)
        return (self.recovery_command_timer > 0).float()

    def _walking_reward_active(self):
        """One after the recovery hold has finished, otherwise zero."""
        if getattr(self.cfg.domain_rand, "recovery_only", False):
            return torch.zeros(self.num_envs, device=self.device)
        return (self.recovery_command_timer == 0).float()

    def _reward_lin_vel_z_up(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])*torch.clamp(-self.projected_gravity[:,2],0,1)*self._walking_reward_active()
    
    def _reward_ang_vel_xy_up(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)*torch.clamp(-self.projected_gravity[:,2],0,1)*self._walking_reward_active()
    
    def _reward_orientation_up(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)*torch.clamp(-self.projected_gravity[:,2],0,1)*self._walking_reward_active()

    def _reward_base_height_up(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)*torch.clamp(-self.projected_gravity[:,2],0,1)*self._walking_reward_active()

    def _reward_base_height_low(self):
        base_height = self._get_base_heights()
        # A low base is unavoidable while feet-up. Start regulating height only
        # after the body has crossed onto its upright side. Once upright, use a
        # symmetric error so crouching and over-extending are both penalized.
        upright_side = torch.clamp(-self.projected_gravity[:, 2], 0., 1.)
        return (
            torch.square(base_height - self.cfg.rewards.base_height_target)
            * upright_side
            * self._recovery_reward_active()
        )

    def _reward_base_height_progress(self):
        base_height = self._get_base_heights()
        height_progress = torch.clamp(base_height / self.cfg.rewards.base_height_target, 0., 1.)
        upright_progress = torch.clamp((1. - self.projected_gravity[:, 2]) * 0.5, 0., 1.)
        return height_progress * (0.25 + 0.75 * upright_progress) * self._recovery_reward_active()

    def _reward_recovery_upright_gaussian(self):
        """FR-Net-style sharp bonus for reaching a genuinely upright base."""
        gravity_error = self.projected_gravity[:, 2] + 1.
        epsilon = self.cfg.domain_rand.recovery_upright_gaussian_width
        return torch.exp(
            -torch.square(gravity_error) / (2. * epsilon * epsilon)
        ) * self._recovery_reward_active()

    def _reward_recovery_target_pose(self):
        """FR-Net step-2 pose reward, active only after becoming upright."""
        gravity_z = self.projected_gravity[:, 2]
        upright_gate = (torch.abs(gravity_z + 1.) <= 0.2).float()

        curriculum_steps = max(
            1, int(getattr(
                self.cfg.domain_rand, "recovery_pose_curriculum_steps", 36000
            ))
        )
        progress = min(float(self.common_step_counter) / curriculum_steps, 1.)
        target = self.recovery_curl_dof_pos + progress * (
            self.recovery_stand_dof_pos - self.recovery_curl_dof_pos
        )
        pose_error = torch.sum(torch.square(self.dof_pos - target), dim=1)
        pose_width = self.cfg.domain_rand.recovery_pose_width
        pose_score = torch.exp(-pose_error / pose_width)
        return pose_score * upright_gate * self._recovery_reward_active()

    def _recovery_stand_gate(self):
        """Smoothly enable settling objectives only after recovery is nearly done.

        Orientation progress is active until upright reaches about 0.85.  This
        gate begins there, avoiding an overlap in which stand-pose and action
        smoothing objectives suppress the final decisive rollover motion.
        """
        upright = -self.projected_gravity[:, 2]
        upright_start = float(getattr(
            self.cfg.domain_rand, "recovery_stand_gate_upright_start", 0.85
        ))
        upright_full = float(getattr(
            self.cfg.domain_rand, "recovery_stand_gate_upright_full", 0.95
        ))
        upright_gate = torch.clamp(
            (upright - upright_start)
            / max(upright_full - upright_start, 1e-6), 0., 1.
        )

        height_ratio = (
            self._get_base_heights()
            / max(float(self.cfg.rewards.base_height_target), 1e-6)
        )
        height_start = float(getattr(
            self.cfg.domain_rand, "recovery_stand_gate_height_start", 0.75
        ))
        height_full = float(getattr(
            self.cfg.domain_rand, "recovery_stand_gate_height_full", 0.90
        ))
        height_gate = torch.clamp(
            (height_ratio - height_start)
            / max(height_full - height_start, 1e-6), 0., 1.
        )
        return upright_gate * height_gate

    def _reward_recovery_stability(self):
        """Penalize residual base/joint motion only after nearly upright."""
        stand_gate = self._recovery_stand_gate()
        motion = torch.sum(torch.square(self.base_lin_vel), dim=1)
        motion += 0.25 * torch.sum(torch.square(self.base_ang_vel), dim=1)
        # The base can look numerically stable while the long legs keep making
        # visible corrections.  A small joint-velocity component suppresses
        # that post-recovery tremor; stand_gate keeps it completely out of the
        # aggressive rollover phase.
        motion += 0.05 * torch.sum(torch.square(self.dof_vel), dim=1)
        return motion * stand_gate * self._recovery_reward_active()

    def _reward_recovery_terminal_success(self):
        """One-off success bonus, discounted when feet-up recovery needed retries."""
        retry_discount = 1. / (1. + self.recovery_retry_count.float())
        return self.recovery_success_buf.float() * retry_discount

    def _get_recovery_leg_surface_clearances(self):
        """Return minimum approximate surface clearance for four front/rear pairs."""
        thigh_states = self.rigid_body_states[:, self.recovery_thigh_indices, :]
        calf_states = self.rigid_body_states[:, self.recovery_calf_indices, :]

        def local_points_to_world(states, local_points):
            if local_points.dim() == 2:
                num_points = local_points.shape[0]
                points = local_points.view(1, 1, num_points, 3).expand(
                    self.num_envs, states.shape[1], -1, -1
                )
            else:
                num_points = local_points.shape[1]
                points = local_points.unsqueeze(0).expand(
                    self.num_envs, -1, -1, -1
                )
            quat = states[:, :, 3:7].unsqueeze(2).expand(
                -1, -1, num_points, -1
            )
            rotated = quat_apply(
                quat.reshape(-1, 4), points.reshape(-1, 3)
            ).view(self.num_envs, states.shape[1], num_points, 3)
            return rotated + states[:, :, 0:3].unsqueeze(2)

        thigh_points = local_points_to_world(
            thigh_states, self.recovery_thigh_local_points
        )
        calf_points = local_points_to_world(
            calf_states, self.recovery_calf_local_points
        )
        leg_points = torch.cat((thigh_points, calf_points), dim=2)
        pair_a = self.recovery_leg_pair_indices[:, 0]
        pair_b = self.recovery_leg_pair_indices[:, 1]
        # [env, four front/rear leg pairs, points on A, points on B]
        center_distances = torch.norm(
            leg_points[:, pair_a, :, None, :]
            - leg_points[:, pair_b, None, :, :],
            dim=-1,
        )
        surface_clearances = center_distances - (
            self.recovery_leg_point_radii.unsqueeze(3)
            + self.recovery_leg_point_radii.unsqueeze(2)
        )
        min_distance = torch.amin(surface_clearances, dim=(2, 3))
        return min_distance

    def _reward_recovery_leg_clearance(self):
        """Steer front/rear legs apart, with an extra cost at contact."""
        min_distance = self._get_recovery_leg_surface_clearances()
        self.recovery_episode_min_leg_clearance[:] = torch.minimum(
            self.recovery_episode_min_leg_clearance, min_distance.detach()
        )
        safe_distance = max(float(getattr(
            self.cfg.domain_rand, "recovery_leg_clearance_distance", 0.07
        )), 1e-6)
        violation = torch.clamp(
            (safe_distance - min_distance) / safe_distance, 0.0, 1.0
        )
        contact_distance = max(float(getattr(
            self.cfg.domain_rand, "recovery_leg_contact_distance", 0.005
        )), 1e-6)
        contact = torch.clamp(
            (contact_distance - min_distance) / contact_distance, 0.0, 1.0
        )
        contact_multiplier = float(getattr(
            self.cfg.domain_rand, "recovery_leg_contact_multiplier", 5.0
        ))
        return (
            torch.sum(
                torch.square(violation) + contact_multiplier * contact,
                dim=1,
            )
            * self._recovery_reward_active()
        )

    def _reward_recovery_leg_contact(self):
        """Strongly penalize actual/near front-rear leg surface contact."""
        min_distance = self._get_recovery_leg_surface_clearances()
        self.recovery_episode_min_leg_clearance[:] = torch.minimum(
            self.recovery_episode_min_leg_clearance, min_distance.detach()
        )
        contact_distance = max(float(getattr(
            self.cfg.domain_rand, "recovery_leg_contact_distance", 0.01
        )), 1e-6)
        # Zero outside 1 cm, one at contact or penetration. This is deliberately
        # much stronger than the 8 cm anticipatory shaping term.
        contact = torch.clamp(
            (contact_distance - min_distance) / contact_distance, 0.0, 1.0
        )
        return torch.sum(contact, dim=1) * self._recovery_reward_active()

    def _reward_recovery_escape(self):
        """Reward leaving the locally symmetric feet-up pose during recovery."""
        recovery_active = (
            (self.recovery_command_timer > 0) & self.recovery_upside_down_active
        )
        # feet-up: 0, side-lying: sqrt(0.5), fully upright: 1.  Unlike the
        # previous version, this remains increasing after the robot reaches its
        # side, so rocking partway over cannot collect the maximum reward.
        escape = torch.sqrt(torch.clamp(
            (1. - self.projected_gravity[:, 2]) * 0.5, 0., 1.
        ))
        return escape * recovery_active

    def _reward_recovery_orientation_progress(self):
        """Reward net rotation toward upright from every fallen orientation."""
        current_gravity_z = self.projected_gravity[:, 2]
        recovery_active = (
            (self._recovery_reward_active() > 0)
            & (current_gravity_z > -0.85)
        )
        progress = self.recovery_last_gravity_z - current_gravity_z
        # Bound rare contact impulses while preserving the sign, so a forward
        # rotation and an equal backward rotation cannot be farmed.  Weight
        # regression more heavily: repeated half-rolls should be worse than one
        # committed roll, while the policy is still free to choose either side.
        progress = torch.clamp(progress, -0.2, 0.2)
        regression_scale = float(getattr(
            self.cfg.domain_rand,
            "recovery_orientation_regression_scale",
            1.0,
        ))
        if getattr(
            self.cfg.domain_rand,
            "recovery_orientation_regression_upside_only",
            False,
        ):
            # A side-lying recovery can require a small initial counter-motion
            # to establish a useful contact. Do not punish that maneuver with
            # the stronger feet-up anti-rocking multiplier.
            effective_regression_scale = torch.where(
                self.recovery_upside_down_active,
                torch.full_like(progress, regression_scale),
                torch.ones_like(progress),
            )
        else:
            effective_regression_scale = regression_scale
        progress = torch.where(
            progress < 0., progress * effective_regression_scale, progress
        ) * recovery_active
        self.recovery_last_gravity_z[recovery_active] = current_gravity_z[recovery_active]
        return progress
    
    def _reward_foot_clearance_up(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        # Clearance is a swing-foot objective. Applying it to stance feet
        # penalizes a correct no-slip stance because those feet move relative
        # to the body while remaining stationary in the world frame.
        no_contact = (~self.contact_filt).float()

        clearance_reward = height_error * foot_leteral_vel * no_contact
        
        return torch.sum(clearance_reward, dim=1)*torch.clamp(-self.projected_gravity[:,2],0,1)*self._walking_reward_active()
    
    def _reward_foot_slide_up(self):
        # A correctly planted stance foot is nearly stationary in the world
        # frame. Using velocity relative to the moving base would penalize that
        # correct behavior and favor feet that slide along with the robot.
        foot_lateral_vel_world = torch.norm(self.feet_vel[:, :, :2], dim=2)
        cost_slide = torch.sum(
            self.contact_filt * foot_lateral_vel_world, dim=1
        ) * torch.clamp(-self.projected_gravity[:, 2], 0, 1) * self._walking_reward_active()
        return cost_slide

    def _reward_stumble_up(self):
        # Penalize feet hitting vertical surfaces

        return torch.clamp(-self.projected_gravity[:,2],0,1)*(torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1))*self._walking_reward_active()
    
    def _reward_collision_up(self):
        # Penalize collisions on selected bodies
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_foot_mirror_up(self):
        diff1 = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[9,10,11]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:,[3,4,5]] - self.dof_pos[:,[6,7,8]]),dim=-1)
        return 0.5*torch.clamp(-self.projected_gravity[:,2],0,1)*(diff1 + diff2)*self._walking_reward_active()
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  100).clip(min=0.), dim=1)

    
    def _reward_upward(self):
        return (1 - self.projected_gravity[:,2]) * self._recovery_reward_active()

    def _reward_stand_success(self):
        base_height = self._get_base_heights()
        upright = torch.clamp(-self.projected_gravity[:, 2], 0, 1)
        contact_count = torch.sum(1. * self.contact_filt, dim=-1)
        height_ok = base_height > (
            self.cfg.domain_rand.recovery_success_height_ratio
            * self.cfg.rewards.base_height_target
        )
        upright_ok = upright > self.cfg.domain_rand.recovery_success_upright
        contact_ok = contact_count >= self.cfg.domain_rand.recovery_success_contacts
        # A geometrically correct pose is not yet a successful recovery if the
        # robot keeps walking, sliding or rocking. Use smooth velocity gates so
        # the reward supplies a gradient all the way toward zero motion, while
        # the hard thresholds in check_termination remain the final criterion.
        lin_speed = torch.norm(self.base_lin_vel, dim=1)
        ang_speed = torch.norm(self.base_ang_vel, dim=1)
        lin_scale = max(
            float(self.cfg.domain_rand.recovery_success_max_lin_vel), 1e-6
        )
        ang_scale = max(
            float(self.cfg.domain_rand.recovery_success_max_ang_vel), 1e-6
        )
        lin_score = torch.exp(-torch.square(lin_speed / lin_scale))
        ang_score = torch.exp(-torch.square(ang_speed / ang_scale))
        # A geometrically upright robot should not receive the full stand
        # reward while its legs remain in a strongly crouched/asymmetric pose.
        # This is deliberately smooth: nearby poses are accepted, while large
        # deviations from the deployment standing reference lose reward.
        pose_rmse = torch.sqrt(torch.mean(torch.square(
            self.dof_pos - self.default_dof_pos
        ), dim=1) + 1e-8)
        pose_width = max(
            float(self.cfg.domain_rand.recovery_pose_score_width), 1e-6
        )
        pose_score = torch.exp(-torch.square(pose_rmse / pose_width))
        return (
            height_ok.float()
            * upright_ok.float()
            * contact_ok.float()
            * lin_score
            * ang_score
            * pose_score
            * self._recovery_reward_active()
        )

    def _reward_stand_ready(self):
        base_height = self._get_base_heights()
        height_progress = torch.clamp(base_height / self.cfg.rewards.base_height_target, 0., 1.)
        upright_progress = torch.clamp(-self.projected_gravity[:, 2], 0., 1.)
        contact_fraction = torch.clamp(torch.sum(1. * self.contact_filt, dim=-1) / 4., 0., 1.)
        # Contact fraction is recovery shaping only. Leaving it active while
        # walking rewards keeping all four feet down and promotes dragging.
        return height_progress * upright_progress * contact_fraction * self._recovery_reward_active()

    def _reward_feet_below_base(self):
        feet_below_base = self.root_states[:, 2].unsqueeze(1) - self.feet_pos[:, :, 2]
        below_score = torch.mean(
            torch.clamp(
                feet_below_base / self.cfg.rewards.base_height_target, 0., 1.
            ),
            dim=1,
        )
        # Keeping the feet below the body helps recovery, but during walking it
        # directly conflicts with lifting a swing foot for ground clearance.
        return below_score * self._recovery_reward_active()
    
    def _reward_has_contact(self):
        contact_filt = 1.*self.contact_filt
        return torch.sum(contact_filt,dim=-1)/4 * self._recovery_reward_active()
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_nice(self):
        stand_gate = self._recovery_stand_gate()
        # Squared error has a vanishing gradient at the target.  The previous
        # L1 error kept pushing with constant magnitude near the default pose
        # and encouraged visible back-and-forth joint corrections.
        return torch.sum(
            torch.square(self.dof_pos - self.default_dof_pos), dim=1
        ) * stand_gate * self._recovery_reward_active()
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_powers(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)
        #return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)

    def _reward_powers_dist(self):
        # Penalize power dist
        return torch.var(self.torques*self.dof_vel, dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        action_change = torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )
        if getattr(self.cfg.domain_rand, "recovery_only", False):
            # Recovery needs fast, non-smooth actions while rolling over. Apply
            # action-rate regularization only after it is nearly upright and
            # has regained enough height to begin the settling phase.
            stand_gate = self._recovery_stand_gate()
            return (
                action_change
                * stand_gate
                * self._recovery_reward_active()
            )
        return action_change * self._walking_reward_active()

    def _reward_recovery_action_jump(self):
        """Penalize only violent recovery action jumps, not useful smooth motion."""
        deadband = float(getattr(
            self.cfg.domain_rand, "recovery_action_jump_deadband", 0.45
        ))
        excess = torch.clamp(
            torch.abs(self.actions - self.last_actions) - deadband, min=0.
        )
        # The ordinary action-rate term takes over near the standing pose. This
        # term is limited to the fallen/rollover phase and leaves all changes
        # inside the deadband completely free.
        fallen_gate = 1. - self._recovery_stand_gate()
        return (
            torch.sum(torch.square(excess), dim=1)
            * fallen_gate
            * self._recovery_reward_active()
        )
    
    def _reward_action_smoothness(self):
        return torch.sum(torch.square(
            self.action_history_buf[:,-1,:] - 2*self.action_history_buf[:,-2,:]+self.action_history_buf[:,-3,:]
        ), dim=1) * self._walking_reward_active()
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma) * self._walking_reward_active()
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) * self._walking_reward_active()

    def _reward_feet_air_time(self):
        # Reward a usable swing phase. The original touchdown-only reward gives
        # no learning signal when a foot remains on the ground, so a dragging
        # gait can become a local optimum. Add a small dense reward while feet
        # are airborne and do not punish short exploratory lift-offs.
        # Use the contact filter prepared once in post_physics_step; updating
        # last_contacts again here would destroy its one-step history.
        contact_filt = self.contact_filt
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt

        touchdown_reward = torch.sum(
            torch.clamp(self.feet_air_time - 0.2, min=0., max=0.3)
            * first_contact,
            dim=1,
        )
        airborne = (~contact_filt).float()
        # Give an immediate signal as soon as a foot leaves the ground. Using
        # accumulated swing progress made the first lift-off signal too small
        # to escape the all-feet-down local optimum. Cap it at two feet so a
        # four-foot jump is not rewarded more than a diagonal gait.
        dense_swing_reward = 0.05 * torch.clamp(
            torch.sum(airborne, dim=1), max=2.
        )

        rew_airTime = touchdown_reward + dense_swing_reward
        rew_airTime *= (
            (torch.norm(self.commands[:, :2], dim=1) > 0.1)
            * self._walking_reward_active()
        )
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    
    def _reward_vertical_contact(self):
        return torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2),dim=-1)
        
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    # def _reward_foot_clearance(self):
    #     foot_height = torch.mean(self.foot_positions[:, :, 2].unsqueeze(1).repeat(1,self.num_height_points,1) - self.measured_heights.unsqueeze(2), dim=1)
    #     foot_xy_vel = torch.norm(self.foot_velocities[:,:,:2],dim=-1)
    #     target_height = 0.1 + 0.02
    #     rew_foot_clearance = torch.sum(torch.square(target_height - foot_height) * foot_xy_vel,dim=-1)
    #     return rew_foot_clearance

     
    # def _reward_foot_clearance(self):
    #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    #     footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
    #     footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     for i in range(len(self.feet_indices)):
    #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
    #         footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
    #     height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
    #     foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
    #     return torch.sum(height_error * foot_leteral_vel, dim=1)

    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        #no_contact = 1.*(self.contact_filt == 0)

        clearance_reward = height_error * foot_leteral_vel 
        
        return torch.sum(clearance_reward, dim=1)
    
    def _reward_foot_slide(self):
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        
        cost_slide = torch.sum(self.contact_filt * foot_leteral_vel, dim=1)
        return cost_slide
    
    def _reward_foot_clearance_hippos(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        hip_pos_scale = (1 + torch.abs(self.dof_pos[:, [0, 3, 6, 9]]))
        return torch.sum(hip_pos_scale * height_error * foot_leteral_vel, dim=1)
    
    def _reward_foot_regular(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        #height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_error = torch.exp(-1*(footpos_in_body_frame[:, :, 2] + self.cfg.rewards.base_height_target)/(0.025*self.cfg.rewards.base_height_target)).view(self.num_envs, -1)
        no_contact = 1.*(self.contact_filt == 0)
        return torch.sum(torch.clamp(height_error,0,1) * no_contact, dim=1)
    
    def _reward_hip_pos(self):
        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        # flag = 1.*(torch.abs(self.commands[:,1]) == 0)
        # return flag * torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
        return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
    
    def _reward_phase_contact(self):
        contact_goal = 1.*(torch.sin(self.phase) > 0.0)
        return torch.mean(torch.abs(1.*self.contact_filt - contact_goal),dim=1)
    
    def _reward_phase_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)

        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_point_flag = 1.*(torch.sin(self.phase) < 0.0)

        return torch.mean(height_point_flag * height_error, dim=1)
    
    def _reward_foot_swing_clearance(self):
        # treat foot as swing when no contact
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        no_contact = 1.*(self.contact_filt == 0)

        return torch.sum(height_error * no_contact, dim=1)
    
    
    # def _reward_foot_clearance(self):
    #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    #     footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
    #     footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     for i in range(len(self.feet_indices)):
    #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
    #         footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
    #     height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
    #     foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)

    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
 
    #     foot_leteral_vel = foot_leteral_vel * (1 + contact_filt)

    #     return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_foot_width_equlity(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        width_1 = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1])
        width_2 = torch.abs(footpos_in_body_frame[:,2,1] - footpos_in_body_frame[:,3,1])

        return 1.*(torch.abs(self.commands[:,1]) == 0)*torch.square(width_1 - width_2)
    
    def _reward_foot_dia_enforce(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        dia_1 = torch.sqrt(torch.sum(torch.square(footpos_in_body_frame[:,0,:] - footpos_in_body_frame[:,2,:]),dim=-1))
        dia_2 = torch.sqrt(torch.sum(torch.square(footpos_in_body_frame[:,1,:] - footpos_in_body_frame[:,3,:]),dim=-1))

        return (torch.square(dia_1 - 0.51) + torch.square(dia_2 - 0.51))/2
    
    def _reward_foot_width_cons(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        width_1 = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1])
        width_2 = torch.abs(footpos_in_body_frame[:,2,1] - footpos_in_body_frame[:,3,1])

        return (torch.square(width_1 - 0.3) + torch.square(width_2 - 0.3))/2.
    
    
    def _reward_hip_pos(self):
        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        flag = 1.*(torch.abs(self.commands[:,1]) == 0)
        return flag * torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
        #return flag * 1.*(torch.abs(torch.sum(self.dof_pos[:, [0, 3, 6, 9]],dim=-1)) > 0.0)

    def _reward_foot_mirror(self):
        diff1 = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[9,10,11]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:,[3,4,5]] - self.dof_pos[:,[6,7,8]]),dim=-1)
        return 0.5*(diff1 + diff2)
    
    def _reward_trot_contact(self):
        contact_filt = 1.*self.contact_filt
        pattern_match1 = torch.mean(torch.abs(contact_filt - self.trot_pattern1),dim=-1)
        pattern_match2 = torch.mean(torch.abs(contact_filt - self.trot_pattern2),dim=-1)
        pattern_match_flag = 1.*(pattern_match1*pattern_match2 > 0)
        return pattern_match_flag*(torch.norm(self.commands[:, :2], dim=1) > 0.1)
    
    #------------ cost functions----------------
    """
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    """
    def _cost_feet_contact_forces(self):
        # penalize high contact forces
        return 1.0*(torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  100).clip(min=0.), dim=1) > 0)

    def _cost_torque_limit(self):
        # constaint torque over limit
        #return 1.*(torch.sum(1.*(torch.abs(self.torques) > self.torque_limits*self.cfg.rewards.soft_torque_limit),dim=1)>0.0)
        # return 1.*(torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)>0.0)
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _cost_pos_limit(self):
        # upper_limit = 1.*(self.dof_pos > self.dof_pos_limits[:, 1])
        # lower_limit = 1.*(self.dof_pos < self.dof_pos_limits[:, 0])
        # out_limit = 1.*(torch.sum(upper_limit + lower_limit,dim=1) > 0.0)
        # return out_limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        # return 1.*(torch.sum(out_of_limits, dim=1)>0.0)
        return torch.sum(out_of_limits, dim=1)
   
    def _cost_dof_vel_limits(self):
        # return 1.*(torch.sum(1.*(torch.abs(self.dof_vel) > self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit),dim=1) > 0.0)
        # return 1.*(torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)>0.0)
         return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _cost_vel_smoothness(self):
        return torch.mean(torch.max(torch.zeros_like(self.dof_vel),torch.abs(self.dof_vel) - (self.dof_vel_limits/2.)),dim=1)
    
    def _cost_acc_smoothness(self):
        acc = (self.last_dof_vel - self.dof_vel) / self.dt
        acc_limit = self.dof_vel_limits/(2.*self.dt)
        return 0.1*torch.mean(torch.max(torch.zeros_like(acc),torch.abs(acc) - acc_limit),dim=1)
    
    def _cost_collision(self):
        return  torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _cost_feet_contact_forces(self):
        # penalize high contact forces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > self.cfg.rewards.max_contact_force), dim=1) > 0.0)
        # return torch.mean(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1))
    
    def _cost_stumble(self):
        # Penalize feet hitting vertical surfaces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2])), dim=1) > 0.0)

    def _cost_base_height(self):
        # Penalize base height away from target
        # base_height = self._get_base_heights()
        # return 1.*(torch.abs(base_height) < self.cfg.rewards.base_height_target) #+ 1.*(torch.abs(base_height) > self.cfg.rewards.base_height_target) 
        # base_height = self._get_base_heights()
        # return torch.square(base_height - self.cfg.rewards.base_height_target)
        base_height = self._get_base_heights()
        # return 1.*(torch.square(base_height - self.cfg.rewards.base_height_target) > 0.0) 
        return 100*torch.square(base_height - self.cfg.rewards.base_height_target)
    
    
    def _cost_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
       
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return torch.max(torch.zeros_like(rew_airTime),-1.*rew_airTime)#1.*(rew_airTime < 0.0)
    
    def _cost_ang_vel_xy(self):
        ang_vel_xy = 0.01*torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return ang_vel_xy
    
    def _cost_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])
    
    def _cost_torques(self):
        # Penalize torques
        torque_squres = 0.0001*torch.sum(torch.square(self.torques),dim=1)
        return torque_squres
    
    def _cost_action_rate(self):
        action_rate = 0.01*torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_rate
    
    def _cost_walking_style(self):
        # number of contact must greater than 2 at each frame
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        return 1.*(torch.sum(1.*contact_filt,dim=-1) < 3.)
    
    def _cost_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_start_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _cost_hip_pos(self):
        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        # return flag * torch.mean(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
        return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - 0.0),dim=-1)
    
    def _cost_feet_height(self):
        # Reward high steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        foot_heights_cost = torch.sum(torch.square(self.dof_pos[:,[2,5,8,11]] - (-2.0)) * (~contact_filt),dim=1)
 
        return foot_heights_cost
    
    def _cost_contact_force_xy(self):
        contact_xy_force_norm = torch.mean(torch.norm(self.contact_forces[:, self.feet_indices, :2],dim=-1),dim=-1)
        return contact_xy_force_norm

    def _cost_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _cost_default_pos(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _cost_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.mean(contact_filt * foot_velocities, dim=1)
        return rew_slip
    
    def _cost_feet_contact_velocity(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        foot_velocities = torch.square(self.foot_velocities[:, :, 2].view(self.num_envs, -1))
        rew_contact_force = torch.mean(contact_filt * foot_velocities, dim=1)
        return rew_contact_force
    
    def _cost_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _cost_foot_swing_clearance(self):
        # treat foot as swing when no contact
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)

        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_error *= ~self.contact_filt

        return 10*torch.sum(height_error, dim=1)
    
    def _cost_foot_swing_clearance_cum(self):
        # treat foot as swing when no contact
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)

        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        no_contact = 1.*(1.*self.contact_filt == 0)

        return torch.mean(torch.abs(footpos_in_body_frame[:, :, 2]) * no_contact, dim=1)
    
    def _cost_foot_slide(self):
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        
        cost_slide = torch.mean(self.contact_filt * foot_leteral_vel, dim=1)
        return cost_slide
    
    def _cost_trot_contact(self):
        contact_filt = 1.*self.contact_filt
        pattern_match1 = torch.mean(torch.abs(contact_filt - self.trot_pattern1),dim=-1)
        pattern_match2 = torch.mean(torch.abs(contact_filt - self.trot_pattern2),dim=-1)
        pattern_match_flag = 1.*(pattern_match1*pattern_match2 > 0)
        return pattern_match_flag*(torch.norm(self.commands[:, :2], dim=1) > 0.1)
    
    def _cost_phase_contact(self):
        contact_goal = 1.*(torch.sin(self.phase) > 0.0)
        return 1.*(torch.mean(torch.abs(1.*self.contact_filt - contact_goal),dim=1) > 0.0)
    
    def _cost_phase_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_point_flag = 1.*(torch.sin(self.phase) < 0.0)

        return torch.sum(height_point_flag* height_error, dim=1)
    
    def _cost_phase_foot_min_height(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        heights = -1*footpos_in_body_frame[:, :, 2]
        height_point_flag = 1.*(torch.sin(self.phase) < 0.0)

        return torch.mean(height_point_flag* heights, dim=1)
    
    def _cost_foot_width(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        width_1 = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1])
        width_2 = torch.abs(footpos_in_body_frame[:,2,1] - footpos_in_body_frame[:,3,1])

        less_width = (1.*(width_1 < 0.28) + 1.*(width_2 < 0.28))/2
        greater_width = (1.*(width_1 > 0.31) + 1.*(width_2 < 0.31))/2

        return (less_width + greater_width)/2

    def _cost_foot_width_equlity(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        width_1 = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1])
        width_2 = torch.abs(footpos_in_body_frame[:,2,1] - footpos_in_body_frame[:,3,1])

        return torch.square(width_1 - width_2)

    def _cost_powers_dist(self):
        # Penalize power dist
        return 10e-5*torch.var(self.torques*self.dof_vel, dim=1)
    
    def _cost_idol_contact(self):
        contact_filt = 1.*self.contact_filt
        sum_contact_filt_flag = 1.*(torch.sum(contact_filt,dim=-1) < 4)
        idol_flag = 1.*(torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return idol_flag*sum_contact_filt_flag
    
    def _cost_idol_hip(self):
        idol_flag = 1.*(torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return idol_flag*torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - 0.0),dim=-1)
    
    def _cost_foot_dia_enforce(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        dia_1 = torch.sqrt(torch.sum(torch.square(footpos_in_body_frame[:,0,:] - footpos_in_body_frame[:,2,:]),dim=-1))
        dia_2 = torch.sqrt(torch.sum(torch.square(footpos_in_body_frame[:,1,:] - footpos_in_body_frame[:,3,:]),dim=-1))

        return (torch.square(dia_1 - 0.51) + torch.square(dia_2 - 0.51))/2
    
    def _cost_foot_regular(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        #height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_error = torch.clamp(torch.exp(footpos_in_body_frame[:, :, 2]/(0.025*self.cfg.rewards.base_height_target)).view(self.num_envs, -1),0,1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _cost_foot_nocontact_regular(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
      
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        #height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_error = torch.clamp(torch.exp(footpos_in_body_frame[:, :, 2]/(0.025*self.cfg.rewards.base_height_target)).view(self.num_envs, -1),0,1)
        height_error *= ~self.contact_filt
        return torch.mean(height_error, dim=1)
    
    def _cost_foot_mirror(self):
        diff1 = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[9,10,11]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:,[3,4,5]] - self.dof_pos[:,[6,7,8]]),dim=-1)
        return 0.05*(diff1 + diff2)
    
    def _cost_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)




    
    
    
    

    
