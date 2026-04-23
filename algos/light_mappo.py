import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.light_mlp_nets import Actor_MLP, Critic_MLP
from utils.scheduler import LinearScheduler
from utils.value_normalizers import create_value_normalizer


class LightMAPPO:

    def __init__(self, args, obs_dim, state_dim, action_dim, device=torch.device("cpu")):
        self.args = args
        self.device = device

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.use_max_grad_norm = args.use_max_grad_norm
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.use_value_norm = args.use_value_norm
        self.lr = args.lr

        # Networks
        self.actor = Actor_MLP(
            obs_dim,
            action_dim,
            args.hidden_size,
            use_feature_normalization=args.use_feature_normalization,
            output_gain=args.actor_gain
        ).to(device)

        self.critic = Critic_MLP(
            state_dim + obs_dim,
            args.hidden_size,
            use_feature_normalization=args.use_feature_normalization
        ).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, eps=args.optimizer_eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=args.optimizer_eps)

        if args.use_linear_lr_decay:
            self.scheduler = LinearScheduler(self.lr, args.min_lr, args.max_steps)

        if self.use_value_norm:
            self.value_normalizer = create_value_normalizer(
                normalizer_type=args.value_norm_type,
                device=device
            )
        else:
            self.value_normalizer = None

    # ==========================================================
    # ACTION SELECTION (FIXED MASKING)
    # ==========================================================

    def get_actions(self, obs, available_actions, deterministic=False):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            available_actions = torch.tensor(available_actions, dtype=torch.float32, device=self.device)

            logits = self.actor(obs)

            # ✅ CRITICAL FIX: clone before masking (avoids in-place issues)
            logits = logits.clone()

            # mask invalid actions
            logits[available_actions == 0] = -1e10

            # use logits directly (more stable)
            dist = torch.distributions.Categorical(logits=logits)

            if deterministic:
                actions = torch.argmax(logits, dim=-1)
                action_log_probs = None
            else:
                actions = dist.sample()
                action_log_probs = dist.log_prob(actions)

            return actions.cpu().numpy(), (
                action_log_probs.cpu().numpy() if action_log_probs is not None else None
            )

    # ==========================================================
    # VALUE FUNCTION
    # ==========================================================

    def get_values(self, state, obs, active_masks):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            active_masks = torch.tensor(active_masks, dtype=torch.float32, device=self.device)

            if state.dim() == 1:
                state = state.unsqueeze(0)

            if state.size(0) != self.args.n_agents:
                state = state.repeat(self.args.n_agents, 1)

            if active_masks.dim() == 1:
                active_masks = active_masks.unsqueeze(1)

            state = state * active_masks

            critic_input = torch.cat((state, obs), dim=-1)
            values = self.critic(critic_input)

            return values.cpu().numpy()

    # ==========================================================
    # UPDATE STEP (FULLY FIXED)
    # ==========================================================

    def update(self, mini_batch):

        (obs_batch, global_state_batch, actions_batch,
         values_batch, returns_batch, masks_batch,
         active_masks_batch, old_action_log_probs_batch,
         advantages_batch, available_actions_batch) = mini_batch

        obs_batch = obs_batch.to(self.device)
        global_state_batch = global_state_batch.to(self.device)
        actions_batch = actions_batch.to(self.device)
        values_batch = values_batch.to(self.device)
        returns_batch = returns_batch.to(self.device)
        old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
        advantages_batch = advantages_batch.to(self.device)
        available_actions_batch = available_actions_batch.to(self.device)

        n_agents = self.args.n_agents

        # =========================
        # FLATTEN
        # =========================
        obs_batch = obs_batch.reshape(-1, obs_batch.shape[-1])
        available_actions_batch = available_actions_batch.reshape(-1, available_actions_batch.shape[-1])
        actions_batch = actions_batch.reshape(-1)

        old_action_log_probs_batch = old_action_log_probs_batch.reshape(-1)
        advantages_batch = advantages_batch.reshape(-1)
        values_batch = values_batch.reshape(-1)
        returns_batch = returns_batch.reshape(-1)

        global_state_batch = global_state_batch.unsqueeze(1).repeat(1, n_agents, 1)
        global_state_batch = global_state_batch.reshape(-1, global_state_batch.shape[-1])

        # =========================
        # FORWARD PASS
        # =========================
        logits = self.actor(obs_batch)
        logits = logits.clone()

        logits[available_actions_batch == 0] = -1e10

        dist = torch.distributions.Categorical(logits=logits)

        action_log_probs = dist.log_prob(actions_batch)
        dist_entropy = dist.entropy()

        critic_input = torch.cat((global_state_batch, obs_batch), dim=-1)
        values = self.critic(critic_input).squeeze(-1)

        # =========================
        # PPO LOSS
        # =========================
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(
            ratio,
            1.0 - self.clip_param,
            1.0 + self.clip_param
        ) * advantages_batch

        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coef * dist_entropy.mean()
        actor_loss = policy_loss + entropy_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        critic_loss = 0.5 * (values - returns_batch).pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

    # ==========================================================
    # TRAIN LOOP
    # ==========================================================

    def train(self, buffer):

        train_info = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy_loss": 0.0,
        }

        for _ in range(self.ppo_epoch):
            mini_batches = buffer.get_minibatches(self.num_mini_batch)

            for mini_batch in mini_batches:
                metrics = self.update(mini_batch)

                for k in train_info:
                    train_info[k] += metrics[k]

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info:
            train_info[k] /= num_updates

        return train_info