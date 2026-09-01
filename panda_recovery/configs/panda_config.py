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
        # Dedicated recovery episode. Every reset samples a curriculum fallen
        # pose; no velocity-tracking samples are mixed into this stage.
        episode_length_s = 12
        
        n_scan = 187
        n_priv_latent = 4 + 1 + 4 + 1 + 1 + 12 + 12 + 12
        n_proprio = 45 + 3
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        use_phase_clock = False
        

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.45] # x,y,z [m]
        reset_lin_vel_range = 0.1
        reset_ang_vel_range = 0.1

        # Preserve the original Panda action space: action == 0 is the 45 cm
        # standing pose. A curled reference is blended in by the controller
        # only while the body is upside down.
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
        stiffness = {'_hip_joint': 30.0,'_thigh_joint': 40.0,'_calf_joint': 65.0}  # [N*m/rad]
        damping = {'_hip_joint': 0.7,'_thigh_joint': 1.0,'_calf_joint': 1.3}     # [N*m*s/rad]

        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.7
        use_filter = True

    class commands( LeggedRobotCfg.commands):
        curriculum = False
        max_forward_curriculum = 0.0
        max_backward_curriculum = 0.0
        max_lat_curriculum = 0.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False
        global_reference = False

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

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
        base_height_target = 0.45
        clearance_height_target = -0.30
        max_contact_force = 100.
        only_positive_rewards = True
        
        class scales( LeggedRobotCfg.rewards.scales ):

            # Existing upright-gated action-rate reward.  Strengthen it only
            # for the short stability fine-tune; recovery actions remain free
            # to change quickly while the body is fallen.
            action_rate = -0.03

            # Locomotion/swing-foot terms are disabled in recovery-only stage.
            foot_clearance_up = 0.0
            foot_mirror_up = 0.0
            foot_slide_up = 0.0
            collision_up = -1.0
            base_height_up = 0.0

            # Dense recovery shaping preserved from the Aug04 run.
            base_height_low = -60.0
            base_height_progress = 4.0
            feet_below_base = 2.0
            # Keep the successful model_3600 shaping, with only a weak signed
            # progress term to reduce repeated feet-up rocking.  The previous
            # 30.0 fine-tune was too strong and changed the recovery strategy.
            recovery_orientation_progress = 10.0
            recovery_upright_gaussian = 0.0
            recovery_target_pose = 0.0
            # True one-step event reward. With dt=0.02 this contributes 5.0
            # once for a first-attempt success and less after every retry.
            recovery_terminal_success = 250.0
            # Active only near upright, so it damps the recovered stand without
            # restricting the aggressive rollover phase.
            recovery_stability = -0.1
            # One geometry evaluation combines a weak approach penalty and a
            # much stronger actual-contact component.
            recovery_leg_clearance = -0.2
            recovery_leg_contact = 0.0
            # Fallen-phase deadband regularizer: normal recovery motion is free,
            # while violent target-position jumps are discouraged.
            recovery_action_jump = -0.15
            stumble_up = -0.05
            # Match the successful historical Panda recovery runs. This also
            # makes rew_upward directly comparable with their logs.
            upward = 3.0
            has_contact = 0.5
            stand_ready = 4.0
            stand_success = 8.0
            # Return the legs near Panda's default standing pose after recovery;
            # the upright gate keeps this term out of the rollover phase.
            stand_nice = -0.5
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z_up = 0.0
            ang_vel_xy_up = 0.0
            orientation_up = 0.0
            feet_contact_forces = -0.00015
            feet_air_time = 0.0
            

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        # Cover the smooth tile used by the real robot as well as rubber/mat
        # contact.  Delayed stand gating and signed orientation progress keep
        # low-friction samples from learning repeated reward-farming rocks.
        friction_range = [0.35, 1.25]
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
        # Every episode starts directly from the time-curriculum fallen pose.
        # Mid-episode pushes are unnecessary for this recovery-only stage.
        recovery_push_robots = False
        recovery_only = True
        recovery_push_interval_s = 8
        recovery_command_hold_s = 3
        recovery_push_lin_vel_xy = 1.2
        recovery_push_ang_vel_xy = 4.0
        recovery_push_ang_vel_z = 1.0
        recovery_randomize_orientation_ratio = 0.25
        recovery_randomize_height_range = [0.25, 0.45]
        # Side-lying recovery plateaued near 65%. Rebalance explicit starts so
        # both sides receive enough updates while 40% feet-up replay preserves
        # the already strong upside-down behavior. The remaining 10% stays as
        # generic randomized fallen poses.
        recovery_upside_down_ratio = 0.40
        recovery_left_side_ratio = 0.25
        recovery_right_side_ratio = 0.25
        recovery_side_roll_noise = 0.15
        recovery_side_pitch_noise = 0.10
        recovery_side_height_range = [0.20, 0.32]
        recovery_upside_down_height_range = [0.22, 0.30]
        # Near-exact feet-up starts retain the hard recovery case, while a very
        # small tilt breaks the perfectly symmetric local minimum during PPO
        # exploration. simple_play still tests an exact 180 degree pose.
        recovery_upside_down_angle_noise = 0.15
        recovery_upside_down_tilt_noise = 0.15
        # Explicitly replay the exact static 180-degree local minimum that the
        # continuous noisy reset distribution otherwise never samples.
        recovery_exact_upside_ratio = 0.60

        # Deployment (rl_sar) always adds actions to the standing reference.
        # Keep training identical so the exported ONNX has the same action
        # semantics in Isaac Gym, MuJoCo and on the real robot.
        use_hybrid_recovery_reference = False
        recovery_reference_blend_start_gz = 0.10
        recovery_reference_blend_full_gz = 0.80
        recovery_curl_hip = 0.0
        recovery_curl_thigh = 1.4
        recovery_curl_calf = -2.3
        # Cover imperfect real-world feet-up leg arrangements instead of
        # resetting every robot to one perfectly symmetric curl.
        recovery_upside_joint_noise = 0.20
        recovery_upside_hip_noise = 0.12
        recovery_upside_random_joint_ratio = 0.25
        recovery_side_joint_noise = 0.25
        recovery_side_hip_noise = 0.20
        recovery_reset_hip_noise = 0.18
        # Surface-to-surface warning distance. Five centimetres gives the
        # policy time to separate the legs without blocking useful compact
        # recovery poses; actual contact receives the stronger component.
        recovery_leg_clearance_distance = 0.05
        recovery_leg_contact_distance = 0.005
        recovery_leg_contact_multiplier = 5.0
        recovery_action_jump_deadband = 0.45
        recovery_retry_escape_gravity_z = 0.25
        recovery_retry_return_gravity_z = 0.65
        recovery_stand_hip = 0.0
        recovery_stand_thigh = 0.65
        recovery_stand_calf = -1.25
        recovery_pose_curriculum_steps = 1
        recovery_pose_width = 0.25
        recovery_upright_gaussian_width = 0.15
        # Backsliding toward feet-up is more costly than equal forward progress.
        # This targets repeated rocking without prescribing a left/right roll.
        recovery_orientation_regression_scale = 1.5
        recovery_orientation_regression_upside_only = True

        # Kept for evaluation metrics only. This recovery-only run
        # does not terminate episodes or advance curriculum from success EMA.
        # A brief upright crossing is not a stable recovery.  This primarily
        # fixes evaluation/curriculum reporting; the dense stand rewards below
        # still train the behavior throughout the rest of the episode.
        recovery_success_hold_s = 1.00
        recovery_success_height_ratio = 0.80
        recovery_success_upright = 0.85
        recovery_success_contacts = 3
        recovery_success_max_lin_vel = 0.30
        recovery_success_max_ang_vel = 0.60
        # Loose pose limits: require a broadly default-like stance without
        # demanding identical joint angles on all four legs.
        recovery_success_pose_rmse = 0.35
        recovery_success_max_joint_error = 0.80
        recovery_pose_score_width = 0.35
        # Do not ask for default pose, low motion or smooth actions during the
        # final rollover.  Settling objectives ramp in only once both body
        # orientation and height indicate that recovery is nearly complete.
        recovery_stand_gate_upright_start = 0.85
        recovery_stand_gate_upright_full = 0.95
        recovery_stand_gate_height_start = 0.75
        recovery_stand_gate_height_full = 0.90
        recovery_terminate_on_success = False
        # Resume the selected stable model_500 with a gradual hard-pose
        # curriculum. The runner restores policy weights but not environment
        # counters, so ramp difficulty over this complete 800-update run.
        # The resumed policy has completed the tilt curriculum. Keep every
        # update at full recovery difficulty during this long fine-tune instead
        # of restarting from easier orientations.
        recovery_curriculum = False
        recovery_curriculum_success_gated = False
        recovery_curriculum_stage_progress = [
            0.00, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00
        ]
        recovery_curriculum_success_threshold = 0.30
        recovery_curriculum_success_ema_alpha = 0.05
        recovery_curriculum_min_stage_steps = 6000
        # The resumed policy is already mature; keep most hard starts from the
        # beginning and finish the ramp over this complete 500-update run.
        recovery_curriculum_steps = 12000
        recovery_curriculum_hard_start = 0.0
        # Begin at 80% of the final explicit hard-pose ratios, then smoothly
        # reach their full values without returning to an easy curriculum.
        recovery_curriculum_initial_hard_ratio = 0.80
        recovery_curriculum_initial_orientation_ratio = 0.05
        recovery_curriculum_initial_tilt = 2.4
        recovery_curriculum_final_tilt = 3.1416
        recovery_curriculum_initial_height_range = [0.38, 0.48]

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
        # Low-rate structural fine-tune from the selected balanced model_1000.
        learning_rate = 1e-5
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
        run_name = 'flat_model1000_exact_upside_safe_first_attempt_finetune'
        experiment_name = 'panda_recovery_0815'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        # This runner restores weights/optimizer but intentionally resets the
        # displayed iteration counter, so this is exactly 800 new updates.
        max_iterations = 8000
        num_steps_per_env = 24
        resume = True
        resume_path = 'logs/panda_recovery_ppo/Aug15_14-16-17_flat_model500_full_difficulty_balanced_recovery_1500/model_1000.pt'


Pandas3RoughCfgPPO = Panda3RoughCfgPPO
