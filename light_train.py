import argparse
import torch
import wandb

from runners.light_mappo_runner import LightMAPPORunner
from runners.light_rnn_mappo_runner import LightRMAPPORunner


def parse_args():
    parser = argparse.ArgumentParser("MAPPO for StarCraft")

    # =================================================
    # Algorithm
    # =================================================

    parser.add_argument("--algo", type=str, default="mappo",
                        choices=["mappo", "mappo_rnn"])

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--cuda", action="store_true", default=False)

    parser.add_argument("--max_steps", type=int, default=10_000_000)


    # =================================================
    # Environment
    # =================================================

    parser.add_argument("--env_name", type=str, default="smacv2",
                        choices=["smacv1", "smacv2"])

    parser.add_argument("--map_name", type=str, default="terran_5_vs_5")

    parser.add_argument("--difficulty", type=str, default="7")

    parser.add_argument("--obs_last_actions", action="store_true", default=False)


    # =================================================
    # Agent settings
    # =================================================

    parser.add_argument("--use_agent_id", action="store_false", default=True)

    parser.add_argument("--use_death_masking", action="store_true", default=False)


    # =================================================
    # Optimizer
    # =================================================

    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--optimizer_eps", type=float, default=1e-5)

    parser.add_argument("--use_linear_lr_decay", action="store_true")

    parser.add_argument("--min_lr", type=float, default=1e-5)


    # =================================================
    # Network
    # =================================================

    parser.add_argument("--hidden_size", type=int, default=128)

    parser.add_argument("--rnn_layers", type=int, default=1)

    parser.add_argument("--data_chunk_length", type=int, default=10)

    parser.add_argument("--fc_layers", type=int, default=2)

    parser.add_argument("--actor_gain", type=float, default=0.01)


    parser.add_argument("--use_feature_normalization", action="store_false", default=True)

    parser.add_argument("--use_value_norm", action="store_true", default=False)

    parser.add_argument("--value_norm_type", type=str, default="welford",
                        choices=["welford", "ema"])

    parser.add_argument("--use_reward_norm", action="store_false", default=True)

    parser.add_argument("--reward_norm_type", type=str, default="efficient",
                        choices=["efficient", "ema"])


    # =================================================
    # PPO (IMPORTANT CHANGES)
    # =================================================

    parser.add_argument("--n_steps", type=int, default=512)

    parser.add_argument("--ppo_epoch", type=int, default=10)

    parser.add_argument("--clip_param", type=float, default=0.2)

    parser.add_argument("--num_mini_batch", type=int, default=4)

    parser.add_argument("--entropy_coef", type=float, default=0.01)

    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--gae_lambda", type=float, default=0.95)

    parser.add_argument("--max_grad_norm", type=float, default=10.0)

    parser.add_argument("--use_clipped_value_loss", action="store_false", default=True)

    parser.add_argument("--use_gae", action="store_false", default=True)

    parser.add_argument("--use_proper_time_limits", action="store_false", default=True)

    parser.add_argument("--use_max_grad_norm", action="store_false", default=True)

    parser.add_argument("--use_huber_loss", action="store_true", default=False)

    parser.add_argument("--huber_delta", type=float, default=10.0)


    # =================================================
    # Evaluation (important for 10M runs)
    # =================================================

    parser.add_argument("--use_eval", action="store_false", default=True)

    parser.add_argument("--eval_interval", type=int, default=100000)

    parser.add_argument("--eval_episodes", type=int, default=32)

    return parser.parse_args()


def main():

    args = parse_args()

    # =================================================
    # Device
    # =================================================

    if args.cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        args.cuda = False


    print("=============================")
    print(f"Training {args.algo.upper()} for StarCraft")
    print("=============================")
    print(f"Environment: {args.env_name}")
    print(f"Map: {args.map_name}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Seed: {args.seed}")
    print(f"Algorithm: {args.algo}")
    print(f"Device: {device}")
    print("=============================")


    # =================================================
    # Weights & Biases
    # =================================================

    wandb.init(
        project="test-smacv2-data",
        name=f"{args.map_name}_seed{args.seed}",
        config=vars(args)
    )


    # =================================================
    # Runner
    # =================================================

    if args.algo == "mappo":
        runner = LightMAPPORunner(args)

    elif args.algo == "mappo_rnn":
        runner = LightRMAPPORunner(args)

    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")


    runner.run()

    wandb.finish()


if __name__ == "__main__":
    main()