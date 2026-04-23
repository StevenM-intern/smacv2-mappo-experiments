"""
Environment factory for creating StarCraft 2 environments with various wrappers.
"""

import platform
import multiprocessing as mp
import random
import numpy as np
import torch

from envs.env_vectorization import SubprocVecEnv, DummyVecEnv


# -------------------------------------------------------------------
# Multiprocessing configuration (CRITICAL FIX)
# -------------------------------------------------------------------
# Windows does NOT support "fork"
# Linux/macOS benefit from "fork"
# This must be set BEFORE any subprocesses are created
# -------------------------------------------------------------------

if mp.get_start_method(allow_none=True) is None:
    if platform.system() != "Windows":
        mp.set_start_method("fork", force=True)
    # On Windows: do nothing (default = spawn)


# -------------------------------------------------------------------
# Single environment creation (MAPPO / SMAC v1)
# -------------------------------------------------------------------

def create_env(args, is_eval=False):
    """
    Create a StarCraft II environment (SMACv1 or SMACv2).
    """
    from envs.wrappers import AgentIDWrapper, DeathMaskingWrapper

    if args.env_name == "smacv1":
        from smac.env import StarCraft2Env

        env = StarCraft2Env(
            map_name=args.map_name,
            difficulty=args.difficulty,
            obs_last_action=args.obs_last_actions,
        )

        env = DeathMaskingWrapper(env, use_death_masking=args.use_death_masking)
        env = AgentIDWrapper(env, use_agent_id=args.use_agent_id)

        return env

    elif args.env_name == "smacv2":
        from envs.smacv2 import SMACv2Env

        env = SMACv2Env(args, seed=args.seed)

        return env

    else:
        raise ValueError(f"Unknown environment name: {args.env_name}")


# -------------------------------------------------------------------
# Environment thunk (for vectorized environments)
# -------------------------------------------------------------------

def make_env(args, base_seed, rank, is_eval=False):
    env_seed = base_seed + rank * 10000

    def _thunk():
        random.seed(env_seed)
        np.random.seed(env_seed)
        torch.manual_seed(env_seed)

        env_name = args.env_name

        if env_name == "smacv1":
            from smac.env import StarCraft2Env

            env = StarCraft2Env(
                map_name=args.map_name,
                seed=env_seed,
                difficulty=args.difficulty,
                obs_last_action=args.obs_last_actions,
            )

            from envs.wrappers import AgentIDWrapper, DeathMaskingWrapper
            env = DeathMaskingWrapper(env, use_death_masking=args.use_death_masking)
            env = AgentIDWrapper(env, use_agent_id=args.use_agent_id)

        elif env_name == "smacv2":
            from envs.smacv2 import SMACv2Env

            # Patch random.shuffle incompatibility (pysc2 bug)
            _orig_shuffle = random.shuffle

            def _patched_shuffle(seq, *args, **kwargs):
                return _orig_shuffle(seq)

            random.shuffle = _patched_shuffle

            env = SMACv2Env(args, seed=env_seed)

        else:
            raise ValueError(f"Unknown environment name: {env_name}")

        return env

    return _thunk


# -------------------------------------------------------------------
# Vectorized environments
# -------------------------------------------------------------------

def make_vec_envs(args, num_processes, is_eval=False):
    base_seed = args.seed + 1_000_000 if is_eval else args.seed

    envs = [
        make_env(args, base_seed, rank=i, is_eval=is_eval)
        for i in range(num_processes)
    ]

    if len(envs) == 1:
        return DummyVecEnv(envs)

    return SubprocVecEnv(envs)
