import numpy as np
import os
from datetime import datetime
from configs.go2_constraint_him import Go2ConstraintHimRoughCfg, Go2ConstraintHimRoughCfgPPO
from configs.panda_config import Panda3RoughCfg, Panda3RoughCfgPPO


import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from utils.task_registry import task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        # Stagger the 4096 environments across the episode horizon. This is
        # the original project behaviour and ensures that every PPO iteration
        # receives freshly completed episode summaries instead of all recovery
        # environments timing out together about once every 25 iterations.
        init_at_random_ep_len=True,
    )

if __name__ == '__main__':
    task_registry.register("go2N3poHim",LeggedRobot,Go2ConstraintHimRoughCfg(),Go2ConstraintHimRoughCfgPPO())
    task_registry.register("pandaN3poHim", LeggedRobot, Panda3RoughCfg(), Panda3RoughCfgPPO())
  
    args = get_args()
    train(args)
