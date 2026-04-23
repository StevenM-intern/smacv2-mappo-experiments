from smacv2.env.starcraft2 import StarCraft2Env as SMACv2Base


class SMACv2Env:
    def __init__(self, args, seed):
        self.env = SMACv2Base(
            map_name=args.map_name,
            seed=seed,
        )

    def get_env_info(self):
        return {
            "n_actions": self.env.get_total_actions(),
            "n_agents": self.env.n_agents,
            "state_shape": self.env.get_state_size()[0],
            "obs_shape": self.env.get_obs_size()[0],
            "episode_limit": self.env.episode_limit,
        }

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def close(self):
        self.env.close()