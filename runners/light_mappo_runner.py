import os
import time
import numpy as np
import torch
import wandb
import shutil

from buffers.light_rollout_storage import RolloutStorage
from envs import create_env
from algos.light_mappo import LightMAPPO
from utils.reward_normalization import EfficientStandardNormalizer, EMANormalizer


class LightMAPPORunner:

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        self.env = create_env(args, is_eval=False)
        self.evaluate_env = create_env(args, is_eval=True)

        env_info = self.env.get_env_info()

        args.n_agents = env_info["n_agents"]
        args.action_dim = env_info["n_actions"]
        args.state_dim = env_info["state_shape"]
        args.obs_dim = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]

        print(f"n_agents: {args.n_agents}, action_dim: {args.action_dim}, state_dim: {args.state_dim}, obs_dim: {args.obs_dim}, episode_limit: {args.episode_limit}")

        self.agent = LightMAPPO(args, args.obs_dim, args.state_dim, args.action_dim, self.device)

        self.buffer = RolloutStorage(
            args.n_steps,
            args.n_agents,
            args.obs_dim,
            args.action_dim,
            args.state_dim
        )

        if args.use_reward_norm:
            self.reward_norm = EMANormalizer() if args.reward_norm_type == "ema" else EfficientStandardNormalizer()

        self.total_steps = 0
        self.episode_rewards = 0
        self.episode_length = 0
        self.episodes = 0

    def clean_sc2_temp(self):
        temp_path = os.path.expanduser(r"~\AppData\Local\Temp\StarCraft II")
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)

    def save_checkpoint(self):
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/{self.args.map_name}_seed{self.args.seed}.pt"

        torch.save({
            "actor": self.agent.actor.state_dict(),
            "critic": self.agent.critic.state_dict(),
            "optimizer_actor": self.agent.actor_optimizer.state_dict(),
            "optimizer_critic": self.agent.critic_optimizer.state_dict(),
            "step": self.total_steps
        }, path)

        print(f"✅ Saved checkpoint @ {self.total_steps}")

    def load_checkpoint(self):
        path = f"checkpoints/{self.args.map_name}_seed{self.args.seed}.pt"

        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)

            self.agent.actor.load_state_dict(checkpoint["actor"])
            self.agent.critic.load_state_dict(checkpoint["critic"])
            self.agent.actor_optimizer.load_state_dict(checkpoint["optimizer_actor"])
            self.agent.critic_optimizer.load_state_dict(checkpoint["optimizer_critic"])

            self.total_steps = checkpoint.get("step", 0)

            print(f"🔁 Loaded checkpoint @ {self.total_steps}")

    def safe_reset_env(self, is_eval=False):
        try:
            if is_eval:
                self.evaluate_env.reset()
            else:
                self.env.reset()
        except Exception as e:
            print("⚠️ RESET FAILED → cleaning temp + restarting:", e)

            self.clean_sc2_temp()

            try:
                if is_eval:
                    self.evaluate_env.close()
                else:
                    self.env.close()
            except:
                pass

            time.sleep(3)

            if is_eval:
                self.evaluate_env = create_env(self.args, is_eval=True)
                self.evaluate_env.reset()
            else:
                self.env = create_env(self.args, is_eval=False)
                self.env.reset()

    def run(self):

        self.load_checkpoint()
        self.warmup()

        evaluate_num = -1
        crash_count = 0

        while self.total_steps < self.args.max_steps:

            try:
                if self.total_steps // self.args.eval_interval > evaluate_num:
                    self.evaluate(self.args.eval_episodes)
                    evaluate_num += 1

                steps = self.collect_rollouts()
                self.total_steps += steps

                self.compute_returns()

                train_info = self.agent.train(self.buffer)
                wandb.log(train_info, step=self.total_steps)

                self.buffer.after_update()

                if self.total_steps % 200000 < steps:
                    self.save_checkpoint()

                if self.total_steps % 2000000 < steps:
                    print("🔄 FULL ENV RESTART")
                    self.clean_sc2_temp()
                    self.env.close()
                    self.evaluate_env.close()
                    time.sleep(3)
                    self.env = create_env(self.args, is_eval=False)
                    self.evaluate_env = create_env(self.args, is_eval=True)

            except Exception as e:
                crash_count += 1
                print(f"💥 CRASH {crash_count}:", e)

                self.save_checkpoint()

                if crash_count > 5:
                    print("❌ TOO MANY CRASHES → FULL RESET")
                    self.clean_sc2_temp()
                    self.env.close()
                    self.evaluate_env.close()
                    time.sleep(5)
                    self.env = create_env(self.args, is_eval=False)
                    self.evaluate_env = create_env(self.args, is_eval=True)
                    crash_count = 0

                self.load_checkpoint()

    def warmup(self):
        self.safe_reset_env()

        self.buffer.obs[0] = np.array(self.env.get_obs(), dtype=np.float32)
        self.buffer.global_state[0] = np.array(self.env.get_state(), dtype=np.float32)
        self.buffer.available_actions[0] = np.array(self.env.get_avail_actions(), dtype=np.float32)

    def collect_rollouts(self):

        obs = self.buffer.obs[0]
        state = self.buffer.global_state[0]
        avail_actions = self.buffer.available_actions[0]
        active_masks = self.buffer.active_masks[0]

        for _ in range(self.args.n_steps):

            actions, action_log_probs = self.agent.get_actions(obs, avail_actions, False)
            values = self.agent.get_values(state, obs, active_masks)

            actions = np.array(actions)

            # 🔥 CRITICAL FIX: refresh avail_actions BEFORE masking
            current_avail_actions = np.array(self.env.get_avail_actions())

            for i in range(len(actions)):
                valid_actions = np.where(current_avail_actions[i] == 1)[0]

                if len(valid_actions) == 0:
                    actions[i] = 0
                elif actions[i] not in valid_actions:
                    actions[i] = np.random.choice(valid_actions)

            try:
                reward, dones, infos = self.env.step(actions)
            except Exception as e:
                print("⚠️ TRAIN CRASH:", e)
                self.safe_reset_env()
                continue

            self.episode_rewards += reward
            self.episode_length += 1

            done = np.all(dones)

            if done:
                self.episodes += 1

                win = 1 if infos.get("battle_won", False) else 0

                wandb.log({
                    "train/win_rate": win,
                    "train/reward": self.episode_rewards,
                    "train/length": self.episode_length,
                }, step=self.total_steps)

                self.safe_reset_env()

                self.episode_rewards = 0
                self.episode_length = 0
                active_masks = np.ones_like(dones)
            else:
                active_masks = 1 - dones

            obs = np.array(self.env.get_obs())
            state = np.array(self.env.get_state())
            avail_actions = np.array(self.env.get_avail_actions())

            self.buffer.insert(
                obs=obs.astype(np.float32),
                global_state=state.astype(np.float32),
                actions=actions.astype(np.int32),
                action_log_probs=np.array(action_log_probs, dtype=np.float32),
                values=np.array(values.squeeze(-1), dtype=np.float32),
                rewards=np.array([reward] * self.args.n_agents, dtype=np.float32),
                masks=np.array([1 - done] * self.args.n_agents, dtype=np.float32),
                active_masks=np.array(active_masks, dtype=np.float32),
                truncates=np.zeros(self.args.n_agents, dtype=np.float32),
                available_actions=avail_actions.astype(np.float32),
            )

        return self.args.n_steps

    def compute_returns(self):

        next_value = self.agent.get_values(
            self.buffer.global_state[-1],
            self.buffer.obs[-1],
            self.buffer.active_masks[-1]
        )

        self.buffer.compute_returns_and_advantages(
            next_value.squeeze(-1),
            self.args.gamma,
            self.args.gae_lambda
        )

    def evaluate(self, num_episodes=10):

        rewards = []
        lengths = []
        wins = []

        for _ in range(num_episodes):

            self.safe_reset_env(is_eval=True)

            episode_reward = 0
            episode_length = 0
            done = False

            while not done:

                obs = np.array(self.evaluate_env.get_obs())
                avail_actions = np.array(self.evaluate_env.get_avail_actions())

                actions, _ = self.agent.get_actions(obs, avail_actions, True)
                actions = np.array(actions)

                # 🔥 also fix eval masking
                for i in range(len(actions)):
                    valid_actions = np.where(avail_actions[i] == 1)[0]
                    if len(valid_actions) == 0:
                        actions[i] = 0
                    elif actions[i] not in valid_actions:
                        actions[i] = np.random.choice(valid_actions)

                reward, dones, infos = self.evaluate_env.step(actions)

                episode_reward += reward
                episode_length += 1
                done = np.all(dones)

            win = 1 if infos.get("battle_won", False) else 0

            rewards.append(episode_reward)
            lengths.append(episode_length)
            wins.append(win)

        wandb.log({
            "eval/reward": np.mean(rewards),
            "eval/win_rate": np.mean(wins),
            "eval/length": np.mean(lengths),
        }, step=self.total_steps)

        print(
            f"{self.total_steps}/{self.args.max_steps} Eval → "
            f"Reward: {np.mean(rewards):.2f}, "
            f"Win: {np.mean(wins):.2f}"
        )