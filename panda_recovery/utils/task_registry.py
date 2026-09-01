import os
import json
import shutil
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from envs.vec_env import VecEnv
from runner import OnConstraintPolicyRunner

from global_config import ROOT_DIR, ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from configs import LeggedRobotCfg, LeggedRobotCfgPPO

def _to_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(key): _to_json_serializable(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(val) for val in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _save_run_config(log_dir, task_name, env_cfg, train_cfg, args):
    if log_dir is None:
        return
    os.makedirs(log_dir, exist_ok=True)
    run_config = {
        "task": task_name,
        "args": vars(args),
        "env_cfg": class_to_dict(env_cfg),
        "train_cfg": class_to_dict(train_cfg),
    }
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(_to_json_serializable(run_config), f, indent=2, sort_keys=True, default=str)
    print(f"Saved run config to: {config_path}")
    _save_source_snapshot(log_dir)


def _copy_source_file(src_path, dst_path):
    if os.path.isfile(src_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _save_source_snapshot(log_dir):
    source_dir = os.path.join(log_dir, "source")
    configs_src_dir = os.path.join(ROOT_DIR, "configs")
    configs_dst_dir = os.path.join(source_dir, "configs")

    if os.path.isdir(configs_src_dir):
        os.makedirs(configs_dst_dir, exist_ok=True)
        for filename in os.listdir(configs_src_dir):
            if filename.endswith(".py"):
                _copy_source_file(
                    os.path.join(configs_src_dir, filename),
                    os.path.join(configs_dst_dir, filename),
                )

    for rel_path in (
        "envs/legged_robot.py",
        "train.py",
        "simple_play.py",
        "utils/task_registry.py",
    ):
        _copy_source_file(
            os.path.join(ROOT_DIR, rel_path),
            os.path.join(source_dir, rel_path),
        )

    print(f"Saved source snapshot to: {source_dir}")


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnConstraintPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        _save_run_config(log_dir, name, env.cfg, train_cfg, args)
        train_cfg_dict = class_to_dict(train_cfg)
        runner_class = eval(train_cfg.runner.runner_class_name)
        runner = runner_class(env, train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        # resume = train_cfg.runner.resume
        # if resume:
        #     # load previously trained model
        #     resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        #     print(f"Loading model from: {resume_path}")
        #     runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
