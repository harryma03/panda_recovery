from configs.go2_constraint_him import Go2ConstraintHimRoughCfg, Go2ConstraintHimRoughCfgPPO
import cv2
import os

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse
from envs import LeggedRobot
from modules import *
from utils import  get_args, export_policy_as_jit, task_registry, Logger
from configs import *
from utils.helpers import class_to_dict, get_load_path
from utils.task_registry import task_registry
import numpy as np
import torch
from global_config import ROOT_DIR

from PIL import Image as im

#如果以后想恢复正常直立启动，将：START_UPSIDE_DOWN = False

START_UPSIDE_DOWN = True
UPSIDE_DOWN_HEIGHT = 0.26
# Visual recovery evaluation.  The viewer shows a 3x3 suite instead of one
# lucky/unlucky rollout; set this to 1 to restore the old single-robot view.
RECOVERY_EVAL_NUM_ENVS = 9
USE_RECOVERY_TEST_SUITE = True
PROFILE_POLICY = False


def set_static_upside_down(env):
    """Place every play environment motionless in a feet-up pose."""
    env.root_states[:, :3] = env.env_origins
    env.root_states[:, 2] += UPSIDE_DOWN_HEIGHT

    count = env.num_envs
    roll = torch.full((count,), np.pi, device=env.device)
    zeros = torch.zeros(count, device=env.device)
    env.root_states[:, 3:7] = quat_from_euler_xyz(roll, zeros, zeros)
    env.root_states[:, 7:13] = 0.
    # Match the feet-up training reset while retaining an exact 180-degree
    # orientation for the harder deterministic evaluation.
    env.dof_pos[:] = env.recovery_curl_dof_pos
    env.dof_vel[:] = 0.

    env.commands.zero_()
    env.actions.zero_()
    env.last_actions.zero_()
    env.obs_history_buf.zero_()
    env.action_history_buf.zero_()
    env.contact_buf.zero_()
    env.recovery_upside_down_active[:] = True
    env.recovery_last_gravity_z[:] = 1.
    env.recovery_stable_steps.zero_()
    # Make the hybrid curl/stand PD reference correct on the very first step
    # after teleporting; post_physics_step would otherwise update this later.
    env.base_quat[:] = env.root_states[:, 3:7]
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)

    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))

    # Advance once so contacts, projected gravity and the first observation all
    # correspond to the teleported upside-down state.
    obs, _, _, _, _, _ = env.step(
        torch.zeros(env.num_envs, env.num_actions, device=env.device)
    )
    return obs


def set_recovery_test_suite(env):
    """Place environments in deterministic fallen poses for visual evaluation."""
    count = env.num_envs
    env.root_states[:, :3] = env.env_origins
    env.root_states[:, 2] += 0.28

    # The first nine cases cover exact/perturbed feet-up, both sides and a
    # pitch-over fall.  Repeat the pattern if a larger env count is selected.
    roll_pattern = torch.tensor([
        np.pi, np.pi, np.pi - 0.15,
        -np.pi + 0.15, -0.5 * np.pi, 0.5 * np.pi,
        0.0, 0.75 * np.pi, -0.75 * np.pi,
    ], device=env.device)
    pitch_pattern = torch.tensor([
        0.0, 0.0, 0.10,
        -0.10, 0.08, -0.08,
        np.pi, 0.18, -0.18,
    ], device=env.device)
    case_ids = torch.arange(count, device=env.device) % len(roll_pattern)
    roll = roll_pattern[case_ids]
    pitch = pitch_pattern[case_ids]
    yaw = torch.zeros(count, device=env.device)
    env.root_states[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    env.root_states[:, 7:13] = 0.

    # Cover both the historical curled simple-play start and the ordinary
    # randomized standing-reference DOFs used by the model_3600 training run.
    env.dof_pos[:] = env.default_dof_pos
    curl_count = min(3, count)
    env.dof_pos[:curl_count] = env.recovery_curl_dof_pos
    if count > 6:
        factors = torch.linspace(
            0.80, 1.20, count - 6, device=env.device
        ).unsqueeze(1)
        env.dof_pos[6:] = env.default_dof_pos * factors
    env.dof_vel[:] = 0.

    env.commands.zero_()
    env.actions.zero_()
    env.last_actions.zero_()
    env.obs_history_buf.zero_()
    env.action_history_buf.zero_()
    env.contact_buf.zero_()
    env.recovery_upside_down_active[:] = torch.abs(roll) > 0.70 * np.pi
    env.recovery_stable_steps.zero_()
    env.recovery_success_buf.zero_()
    env.recovery_episode_success.zero_()
    env.base_quat[:] = env.root_states[:, 3:7]
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.recovery_last_gravity_z[:] = env.projected_gravity[:, 2]

    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    obs, _, _, _, _, _ = env.step(
        torch.zeros(env.num_envs, env.num_actions, device=env.device)
    )
    labels = [
        "upside_curl", "upside_curl", "upside_tilt_curl",
        "upside_tilt", "left_side", "right_side",
        "pitch_upside", "left_oblique", "right_oblique",
    ]
    return obs, [labels[i % len(labels)] for i in range(count)]


def set_keyboard_recovery_pose(env, env_id, pose_name):
    """Teleport one selected play robot to a requested recovery-test pose."""
    idx = torch.tensor([env_id], device=env.device, dtype=torch.long)
    env.root_states[idx, :3] = env.env_origins[idx]
    env.root_states[idx, 7:13] = 0.

    if pose_name == "stand":
        roll = torch.zeros(1, device=env.device)
        pitch = torch.zeros(1, device=env.device)
        height = env.cfg.rewards.base_height_target
        env.dof_pos[idx] = env.default_dof_pos
        upside = False
    else:
        height = 0.28
        env.dof_pos[idx] = env.default_dof_pos
        if pose_name == "upside":
            roll = torch.full((1,), np.pi, device=env.device)
            pitch = torch.zeros(1, device=env.device)
            env.dof_pos[idx] = env.recovery_curl_dof_pos
            upside = True
        elif pose_name == "left":
            roll = torch.full((1,), -0.5 * np.pi, device=env.device)
            pitch = torch.zeros(1, device=env.device)
            upside = False
        elif pose_name == "right":
            roll = torch.full((1,), 0.5 * np.pi, device=env.device)
            pitch = torch.zeros(1, device=env.device)
            upside = False
        else:
            roll = (2.0 * torch.rand(1, device=env.device) - 1.0) * np.pi
            pitch = (2.0 * torch.rand(1, device=env.device) - 1.0) * np.pi
            joint_scale = 0.75 + 0.50 * torch.rand(
                1, env.num_dof, device=env.device
            )
            env.dof_pos[idx] = env.default_dof_pos * joint_scale
            upside = False

    env.root_states[idx, 2] += height
    yaw = torch.zeros(1, device=env.device)
    env.root_states[idx, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    env.dof_vel[idx] = 0.
    env.commands[idx] = 0.
    env.actions[idx] = 0.
    env.last_actions[idx] = 0.
    env.obs_history_buf[idx] = 0.
    env.action_history_buf[idx] = 0.
    env.contact_buf[idx] = 0.
    env.recovery_upside_down_active[idx] = upside
    env.recovery_stable_steps[idx] = 0
    env.recovery_success_buf[idx] = False
    env.recovery_episode_success[idx] = False
    env.episode_length_buf[idx] = 0
    env.base_quat[idx] = env.root_states[idx, 3:7]
    env.base_lin_vel[idx] = 0.
    env.base_ang_vel[idx] = 0.
    env.projected_gravity[idx] = quat_rotate_inverse(
        env.base_quat[idx], env.gravity_vec[idx]
    )
    env.recovery_last_gravity_z[idx] = env.projected_gravity[idx, 2]

    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.compute_observations()
    print(f"Keyboard recovery pose: env={env_id}, pose={pose_name}")
    return env.get_observations()

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, RECOVERY_EVAL_NUM_ENVS)
    env_cfg.env.env_spacing = 1.5
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    #env_cfg.terrain.mesh_type = 'plane'
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.recovery_push_robots = False
    #env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.add_noise = False
    env_cfg.commands.resampling_time = 1e9
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    # Keep the control filter from the training configuration.
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_kpkd = False
    env_cfg.domain_rand.randomize_init_state = False
    # Training resets after a persistent successful recovery to collect more
    # attempts. Play keeps the recovered robot standing for visual inspection.
    env_cfg.domain_rand.recovery_terminate_on_success = False
    env_cfg.env.episode_length_s = 1.0e9
    # Base contact is expected during a recovery test and must not reset it.
    env_cfg.asset.terminate_after_contacts_on = []
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    test_labels = ["default"] * env.num_envs
    if START_UPSIDE_DOWN and USE_RECOVERY_TEST_SUITE:
        obs, test_labels = set_recovery_test_suite(env)
        print(f"Recovery test suite enabled: {env.num_envs} parallel cases")
    elif START_UPSIDE_DOWN:
        obs = set_static_upside_down(env)
        print(f"Fixed upside-down start enabled at {UPSIDE_DOWN_HEIGHT:.2f} m")
    else:
        obs = env.get_observations()
    # load policy partial_checkpoint_load
    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)
    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy: ActorCriticRMA = actor_critic_class(env.cfg.env.n_proprio,
                                                      env.cfg.env.n_scan,
                                                      env.num_obs,
                                                      env.cfg.env.n_priv_latent,
                                                      env.cfg.env.history_len,
                                                      env.num_actions,
                                                      **policy_cfg_dict)
    print(policy)
    log_root = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    # With no CLI override, evaluate the exact checkpoint selected by the
    # training configuration.  The old behavior silently picked the newest log
    # directory, which could evaluate model_10000 while the user expected the
    # model_3600 resume baseline.
    configured_path = getattr(train_cfg.runner, "resume_path", "")
    if (args.load_run is None and args.checkpoint is None
            and getattr(train_cfg.runner, "resume", False)
            and configured_path):
        load_path = configured_path
        if not os.path.isabs(load_path):
            load_path = os.path.join(ROOT_DIR, load_path)
    else:
        load_run = args.load_run if args.load_run is not None else -1
        checkpoint = args.checkpoint if args.checkpoint is not None else -1
        load_path = get_load_path(
            log_root, load_run=load_run, checkpoint=checkpoint
        )
    print(f"Loading model from: {load_path}")
    model_dict = torch.load(load_path)
    policy.load_state_dict(model_dict['model_state_dict'])
    policy = policy.to(env.device)
    onnx_path = os.path.join(ROOT_DIR, 'model.onnx')
    policy.save_torch_onnx_policy(onnx_path, env.device)
    print(f"Exported ONNX policy to: {onnx_path}")
    policy.half()
    policy.eval()

    # clear images under frames folder
    # frames_path = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    # delete_files_in_directory(frames_path)

    cam_handle = None
    if RECORD_FRAMES:
        # Body-following recording remains focused on env 0; the interactive
        # viewer shows the complete 3x3 evaluation grid.
        camera_local_transform = gymapi.Transform()
        camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
        camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.deg2rad(90))
        camera_props = gymapi.CameraProperties()
        camera_props.width = 512
        camera_props.height = 512
        cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
        env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)

    img_idx = 0

    # Keep interactive play open long enough for repeated keyboard tests while
    # retaining a short deterministic headless evaluation.
    video_duration = 120 if env.viewer is not None else 20
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None

    #torch.sum(self.last_actions - self.actions, dim=1)
    # self.base_lin_vel[:, 2]
    #torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    action_rate = 0
    z_vel = 0
    xy_vel = 0
    feet_air_time = 0

    env.commands[:, :] = 0.
    print("Keyboard: 0-8 select robot; U upside, J left, K right, R random fall, T stand")
    print("          W/S vx, Q/E vy, A/D heading, X stop, SPACE pause, F free camera")

    if env.viewer is not None and env.num_envs > 1:
        # Overview of the compact evaluation grid in the interactive viewer.
        env.free_cam = True
        env.set_camera([1.5, -5.5, 5.0], [1.5, 1.5, 0.0])

    first_success_time = torch.full(
        (env.num_envs,), float("nan"), device=env.device
    )
    post_success_samples = torch.zeros(env.num_envs, device=env.device)
    post_lin_speed_sum = torch.zeros(env.num_envs, device=env.device)
    post_ang_speed_sum = torch.zeros(env.num_envs, device=env.device)
    post_dof_speed_sum = torch.zeros(env.num_envs, device=env.device)
    post_action_delta_sum = torch.zeros(env.num_envs, device=env.device)
    previous_policy_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device
    )

    for i in range(num_frames):
        action_rate += torch.sum(torch.abs(env.last_actions - env.actions),dim=1)
        z_vel += torch.square(env.base_lin_vel[:, 2])
        xy_vel += torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        actions = policy.act_teacher(obs.half())
        policy_action_delta = torch.mean(
            torch.abs(actions.float() - previous_policy_actions), dim=1
        )
        previous_policy_actions[:] = actions.float()
        # actions = torch.clamp(actions,-1.2,1.2)

        obs, privileged_obs, rewards,costs,dones, infos = env.step(actions)
        recovery_request = getattr(env, "keyboard_recovery_request", None)
        if recovery_request is not None:
            obs = set_keyboard_recovery_pose(
                env, env.lookat_id, recovery_request
            )
            first_success_time[env.lookat_id] = float("nan")
            post_success_samples[env.lookat_id] = 0.
            post_lin_speed_sum[env.lookat_id] = 0.
            post_ang_speed_sum[env.lookat_id] = 0.
            post_dof_speed_sum[env.lookat_id] = 0.
            post_action_delta_sum[env.lookat_id] = 0.
            previous_policy_actions[env.lookat_id] = 0.
            env.keyboard_recovery_request = None
        newly_successful = env.recovery_episode_success & torch.isnan(first_success_time)
        first_success_time[newly_successful] = (i + 1) * env.dt
        stable_mask = env.recovery_episode_success.float()
        post_success_samples += stable_mask
        post_lin_speed_sum += torch.norm(env.base_lin_vel, dim=1) * stable_mask
        post_ang_speed_sum += torch.norm(env.base_ang_vel, dim=1) * stable_mask
        post_dof_speed_sum += torch.mean(
            torch.abs(env.dof_vel), dim=1
        ) * stable_mask
        post_action_delta_sum += policy_action_delta * stable_mask
        if RECORD_FRAMES:
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR).reshape((512,512,4))[:,:,:3]
            if video is None:
                video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1 
    print("action rate mean:", torch.mean(action_rate/num_frames).item())
    print("z vel mean:", torch.mean(z_vel/num_frames).item())
    print("xy_vel mean:", torch.mean(xy_vel/num_frames).item())
    print("feet air reward",feet_air_time/num_frames)

    upright = -env.projected_gravity[:, 2]
    base_height = env._get_base_heights()
    pose_error = torch.mean(
        torch.abs(env.dof_pos - env.default_dof_pos), dim=1
    )
    pose_abs_error = torch.abs(env.dof_pos - env.default_dof_pos)
    pose_rmse = torch.sqrt(
        torch.mean(torch.square(pose_abs_error), dim=1) + 1e-8
    )
    max_joint_error = torch.max(pose_abs_error, dim=1).values
    pose_ok = (
        (pose_rmse < env.cfg.domain_rand.recovery_success_pose_rmse)
        & (max_joint_error
           < env.cfg.domain_rand.recovery_success_max_joint_error)
    )
    final_stable = (
        (upright > env.cfg.domain_rand.recovery_success_upright)
        & (base_height > env.cfg.domain_rand.recovery_success_height_ratio
           * env.cfg.rewards.base_height_target)
        & (torch.norm(env.base_lin_vel, dim=1)
           < env.cfg.domain_rand.recovery_success_max_lin_vel)
        & (torch.norm(env.base_ang_vel, dim=1)
           < env.cfg.domain_rand.recovery_success_max_ang_vel)
        & pose_ok
    )
    print("\nRecovery evaluation summary:")
    for idx, label in enumerate(test_labels):
        time_value = first_success_time[idx].item()
        time_text = f"{time_value:.2f}s" if np.isfinite(time_value) else "failed"
        print(
            f"  env {idx}: {label:18s} first_stable={time_text:>7s} "
            f"final_stable={bool(final_stable[idx])} "
            f"pose_error={pose_error[idx].item():.3f}rad "
            f"pose_rmse={pose_rmse[idx].item():.3f}rad "
            f"max_joint_error={max_joint_error[idx].item():.3f}rad"
        )
    print(
        f"success={env.recovery_episode_success.float().mean().item():.1%}, "
        f"final_stable={final_stable.float().mean().item():.1%}"
    )
    sample_denom = torch.clamp(post_success_samples, min=1.)
    post_lin_speed = post_lin_speed_sum / sample_denom
    post_ang_speed = post_ang_speed_sum / sample_denom
    post_dof_speed = post_dof_speed_sum / sample_denom
    post_action_delta = post_action_delta_sum / sample_denom
    print(
        "post-success means: "
        f"lin_speed={post_lin_speed.mean().item():.4f}m/s, "
        f"ang_speed={post_ang_speed.mean().item():.4f}rad/s, "
        f"dof_speed={post_dof_speed.mean().item():.4f}rad/s, "
        f"action_delta={post_action_delta.mean().item():.5f}"
    )

    if video is not None:
        video.release()

    #test model profile
    if PROFILE_POLICY:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
             for i in range(1000):
                with torch.no_grad():
                  actions = policy.act_teacher(obs.half())
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

if __name__ == '__main__':
    task_registry.register("go2N3poHim",LeggedRobot,Go2ConstraintHimRoughCfg(),Go2ConstraintHimRoughCfgPPO())
    task_registry.register("pandaN3poHim", LeggedRobot, Panda3RoughCfg(), Panda3RoughCfgPPO())
  
    args = get_args()
    RECORD_FRAMES = not args.headless
    play(args)
