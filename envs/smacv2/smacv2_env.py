from __future__ import absolute_import, division, print_function

import numpy as np
from absl import logging
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

logging.set_verbosity(logging.ERROR)

import os.path as osp
import yaml

from gymnasium.spaces import Box, Discrete


class SMACv2Env:

    def __init__(self, args, seed=None):

        self.map_config = self._load_map_config(args.map_name)
        self.use_agent_id = args.use_agent_id
        self.use_death_masking = args.use_death_masking

        self._seed = args.seed
        if self._seed is None:
            raise ValueError("SMACv2Env requires a seed to be set.")

        self.map_config["seed"] = self._seed

        self.env = StarCraftCapabilityEnvWrapper(**self.map_config)

        env_info = self.env.get_env_info()

        n_actions = env_info["n_actions"]
        state_shape = env_info["state_shape"]
        obs_shape = env_info["obs_shape"]

        self.episode_limit = env_info["episode_limit"]
        self.n_agents = env_info["n_agents"]

        self.timeouts = self.env.env.timeouts

        self.share_observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_shape,),
            dtype=np.float32,
        )

        if self.use_agent_id:
            obs_shape_with_id = obs_shape + self.n_agents
            self.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_shape_with_id,),
                dtype=np.float32,
            )
        else:
            self.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_shape,),
                dtype=np.float32,
            )

        self.action_space = Discrete(n_actions)

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self):

        self.env.reset()

        obs = self.env.get_obs()
        state = self.env.get_state()
        available_actions = self.env.get_avail_actions()

        if self.use_agent_id:
            obs = self._add_agent_id_to_obs(obs)

        return obs, state, available_actions

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, actions):

        reward, terminated, info = self.env.step(actions)

        obs = self.env.get_obs()
        state = self.env.get_state()
        available_actions = self.env.get_avail_actions()

        if self.use_agent_id:
            obs = self._add_agent_id_to_obs(obs)

        rewards = [[reward]] * self.n_agents

        info["truncated"] = False

        if terminated:

            dones = [True] * self.n_agents

            if self.env.env.timeouts > self.timeouts:

                assert (
                    self.env.env.timeouts - self.timeouts == 1
                ), "Change of timeouts unexpected."

                info["truncated"] = True
                self.timeouts = self.env.env.timeouts

        elif self.use_death_masking:

            dones = [
                bool(self.env.env.death_tracker_ally[agent_id])
                for agent_id in range(self.n_agents)
            ]

        else:

            dones = [False] * self.n_agents

        info.update(
            {
                "win": self.env.env.win_counted,
                "lost": self.env.env.defeat_counted,
                "battles_game": self.env.env.battles_game,
                "battles_won": self.env.env.battles_won,
                "battle_won": self.env.env.win_counted,
            }
        )

        return obs, state, rewards, dones, info, available_actions

    # ==========================================================
    # SPAWN COUNTING (FOR YOUR EXPERIMENT)
    # ==========================================================

    def get_spawn_counts(self):

        marine = 0
        marauder = 0
        medivac = 0

        try:

            for unit in self.env.env.agents.values():

                if unit.unit_type == self.env.env.marine_id:
                    marine += 1

                elif unit.unit_type == self.env.env.marauder_id:
                    marauder += 1

                elif unit.unit_type == self.env.env.medivac_id:
                    medivac += 1

        except Exception:
            pass

        return marine, marauder, medivac

    # ==========================================================
    # REPLAY SAVE
    # ==========================================================

    def save_replay(self):

        try:
            self.env.save_replay()
        except Exception as e:
            print(f"Replay saving failed: {e}")

    # ==========================================================
    # OTHER METHODS
    # ==========================================================

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def get_obs(self):
        return self.env.get_obs()

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    # ==========================================================
    # OBS HELPERS
    # ==========================================================

    def _add_agent_id_to_obs(self, obs):

        obs = np.asarray(obs, dtype=np.float32)
        eye = np.eye(self.n_agents, dtype=np.float32)

        return np.concatenate([obs, eye], axis=1)

    # ==========================================================
    # LOAD YAML MAP CONFIG
    # ==========================================================

    def _load_map_config(self, map_name):

        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "config", f"{map_name}.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Map config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            map_config = yaml.load(file, Loader=yaml.FullLoader)

        return map_config