from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper


class SMACv2Env:
    def __init__(self, args, seed):

        # =====================================
        # UPDATED DISTRIBUTION (Medivac Heavy)
        # =====================================
        capability_config = {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "weights": [0.25, 0.25, 0.50],  # <-- UPDATED
                "exception_unit_types": ["medivac"],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        }

        self.env = StarCraftCapabilityEnvWrapper(
            capability_config=capability_config,
            map_name="10gen_terran",
            seed=seed,
            debug=False,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
        )

    # ==========================================================
    # Spawn Counter (For Mentor Requirement)
    # ==========================================================

    def get_spawn_counts(self):
        marine = 0
        marauder = 0
        medivac = 0

        for _, unit in self.env.agents.items():
            if unit.unit_type == self.env.marine_id:
                marine += 1
            elif unit.unit_type == self.env.marauder_id:
                marauder += 1
            elif unit.unit_type == self.env.medivac_id:
                medivac += 1

        return marine, marauder, medivac

    # ==========================================================

    def get_env_info(self):
        return self.env.get_env_info()

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def get_obs(self):
        return self.env.get_obs()

    def get_state(self):
        return self.env.get_state()

    def get_stats(self):
        battles_won = self.env.battles_won
        battles_game = self.env.battles_game

        win_rate = battles_won / battles_game if battles_game > 0 else 0.0

        return {
            "battles_won": battles_won,
            "battles_game": battles_game,
            "timeouts": self.env.timeouts,
            "win_rate": win_rate,
        }

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def close(self):
        self.env.close()