# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

#from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Panda3RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        episode_length_s = 20
        
        n_scan = 187
        n_priv_latent = 4 + 1 + 4 + 1 + 1 + 12 + 12 + 12
        n_proprio = 45 + 3
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        use_phase_clock = False
        

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.48] # x,y,z [m]
        reset_lin_vel_range = 0.1
        reset_ang_vel_range = 0.1

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0,   # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.65,     # [rad]
            'RL_thigh_joint': 0.65,   # [rad]
            'FR_thigh_joint': 0.65,     # [rad]
            'RR_thigh_joint': 0.65,   # [rad]

            'FL_calf_joint': -1.25,   # [rad]
            'RL_calf_joint': -1.25,    # [rad]
            'FR_calf_joint': -1.25,  # [rad]
            'RR_calf_joint': -1.25,    # [rad]
            
            # 'FL_thigh_joint': 0.52,     # [rad]
            # 'RL_thigh_joint': 0.52,   # [rad]
            # 'FR_thigh_joint': 0.52,     # [rad]
            # 'RR_thigh_joint': 0.52,   # [rad]

            # 'FL_calf_joint': -1.05,   # [rad]
            # 'RL_calf_joint': -1.05,    # [rad]
            # 'FR_calf_joint': -1.05,  # [rad]
            # 'RR_calf_joint': -1.05,    # [rad]            
            
            
            
        } #45cm

        start_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,

            'FL_thigh_joint': 0.65,     # [rad]
            'RL_thigh_joint': 0.65,   # [rad]
            'FR_thigh_joint': 0.65,     # [rad]
            'RR_thigh_joint': 0.65,   # [rad]

            'FL_calf_joint': -1.25,   # [rad]
            'RL_calf_joint': -1.25,    # [rad]
            'FR_calf_joint': -1.25,  # [rad]
            'RR_calf_joint': -1.25,    # [rad]
            
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 60.0}  # [N*m/rad]
        # damping = {'joint': 2.0}     # [N*m*s/rad]
        # armatures = {'hip': 0.00, 'thigh': 0.00, 'calf': 0.00}     # [kg*m2]

        # stiffness = {'_hip_joint': 45.0,'_thigh_joint': 120.0,'_calf_joint': 160.0}  # [N*m/rad]
        # damping = {'_hip_joint': 1.6,'_thigh_joint': 2.4,'_calf_joint': 6.5}     # [N*m*s/rad]

        stiffness = {'_hip_joint': 30.0,'_thigh_joint': 40.0,'_calf_joint': 65.0}  # [N*m/rad]
        damping = {'_hip_joint': 0.7,'_thigh_joint': 1.0,'_calf_joint': 1.3}     # [N*m*s/rad]

        # stiffness = {'_hip_joint': 38.0,'_thigh_joint': 90.0,'_calf_joint': 130.0}  # [N*m/rad]
        # damping = {'_hip_joint': 2.0,'_thigh_joint': 3.5,'_calf_joint': 7.0}     # [N*m*s/rad] 
        
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.7
        use_filter = True

    class commands( LeggedRobotCfg.commands):
        curriculum = True
        max_forward_curriculum = 2.5
        max_backward_curriculum = 1.5
        max_lat_curriculum = 1.5
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True
        global_reference = False

        class ranges:
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-1.5, 1.5]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = "{ROOT_DIR}/resources/panda3_2/urdf/panda3_2.urdf"
        name = "panda3_2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = []#["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        
        # self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.43
        clearance_height_target = -0.30
        max_contact_force = 100.
        only_positive_rewards = True
        
        class scales( LeggedRobotCfg.rewards.scales ):
            foot_clearance_up = -0.5
            foot_mirror_up = -0.05
            foot_slide_up = -0.05
            collision_up = -1.0
            base_height_up = -25.0
            # Dense shaping for fall recovery.
            base_height_low = -60.0
            base_height_progress = 4.0
            feet_below_base = 2.0
            stumble_up = -0.05
            upward = 3.0
            has_contact = 0.5
            stand_ready = 4.0
            stand_success = 8.0
            recovery_upright = 3.0
            recovery_progress = 20.0
            recovery_motion = 0.25
            upside_escape = 2.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            stand_nice = -0.1
            lin_vel_z_up = -4.0
            ang_vel_xy_up = -0.1
            orientation_up = -0.2
            feet_contact_forces = -0.00015
 
            

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        #friction_range = [0.2, 2.75]
        friction_range = [0.2, 1.25]
        randomize_restitution = False
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        #added_mass_range = [-1., 3.]
        added_mass_range = [-1, 2.]
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1
        # Start with small tilts/few randomized robots, then progress to full
        # upside-down orientations. Commands stay at zero during recovery.
        recovery_push_robots = True
        recovery_push_interval_s = 12
        # Recovery ends after the robot is actually stable, not after a fixed
        # command-hold window. A failed attempt is reset after 6 seconds.
        recovery_command_hold_s = 3
        recovery_max_time_s = 10
        recovery_stable_time_s = 0.2
        recovery_success_height_ratio = 0.80
        recovery_success_upright = 0.85
        recovery_success_min_contacts = 3
        recovery_init_upside_down_ratio = 0.30
        recovery_push_lin_vel_xy = 1.2
        recovery_push_ang_vel_xy = 4.0
        recovery_push_ang_vel_z = 1.0
        recovery_randomize_orientation_ratio = 0.40
        recovery_randomize_height_range = [0.22, 0.32]
        # Thirty percent of all robots are explicitly sampled around a
        # feet-up pose at each recovery event (0.40 * 0.75).
        recovery_upside_down_ratio = 0.75
        recovery_upside_down_angle_noise = 0.15
        recovery_upside_down_tilt_noise = 0.15
        recovery_upside_down_velocity_noise = 0.1
        # Begin with a mixture of 120--180 degree flips, then concentrate on
        # the difficult 170--180 degree region.
        recovery_upside_down_curriculum = True
        recovery_upside_down_curriculum_steps = 72000
        recovery_upside_down_initial_min_angle = 2.10
        recovery_upside_down_final_min_angle = 2.97
        recovery_curriculum = False
        recovery_curriculum_steps = 72000
        recovery_curriculum_initial_orientation_ratio = 0.05
        recovery_curriculum_initial_tilt = 0.7
        recovery_curriculum_final_tilt = 3.1416

        # Every episode also starts from a randomized pose. After standing up,
        # the same episode continues with flat-ground velocity tracking.
        randomize_init_state = True

        randomize_motor = True
        motor_strength_range = [0.9, 1.1]

        randomize_kpkd = True
        kp_range = [0.9,1.1]
        kd_range = [0.9,1.1]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
    
    class depth( LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 1  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True
    
    class costs:
        class scales:
            pos_limit = 0.1
            torque_limit = 0.1
            dof_vel_limits = 0.1
            # vel_smoothness = 0.1
            # acc_smoothness = 0.05
            #collision = 0.1
            # feet_contact_forces = 0.1
            # stumble = 0.1
            #feet_air_time = 1
            #torques= 1
            #action_rate= 1
            #base_height=1
            # stand_still=1
            # hip_pos=0.3
 
        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            # vel_smoothness = 0.0
            # acc_smoothness = 3.0
            #collision = 0.0
            # feet_contact_forces = 0.02
            # stumble = 0.0
            #feet_air_time = 0.0
            #torques = 0.025
            #action_rate=0.07
            #base_height=0.0
            # stand_still=0.0
            #hip_pos=0.0
    
 
    
    class cost:
        num_costs = 3
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = True
        include_act_obs_pair_buf = False
        


class Panda3RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1e-3
        max_grad_norm = 1
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 1
        cost_viol_loss_coef = 1

    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = None
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 3

        teacher_act = True
        imi_flag = True
      
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'flat_fast_upside_recovery'
        experiment_name = 'flat_panda_constraint'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 20000
        num_steps_per_env = 24
        resume = True
        resume_path = '/home/user/harryma/can/LocomotionWithNP3O-dev/logs/flat_panda_constraint/Aug04_17-58-13_flat_fast_upside_recovery/model_20000.pt'


Pandas3RoughCfgPPO = Panda3RoughCfgPPO
