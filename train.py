"""
Main training entry point.

Usage:
python train.py                         # Default config, 500k steps
python train.py --config high_lr        # Alternative hyperparams
python train.py --timesteps 1000000     # Longer run
python train.py --no-wandb              # Train without W&B (offline)
"""

import argparse
import os
import sys

print("=" * 60)
print("Python executable:")
print(sys.executable)
print("=" * 60)

from dotenv import load_dotenv
import wandb
import yaml

from agent import PricingAgent

load_dotenv()  # Reads .env file for WANDB_API_KEY


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pricing RL agent")

    parser.add_argument("--config", default="default", help="Policy config name")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")

    # Hyperparameters supplied by W&B Sweep
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--clip_range", type=float, default=None)
    parser.add_argument("--ent_coef", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    use_wandb = (
        not args.no_wandb
        and os.getenv("WANDB_API_KEY") is not None
    )

    if use_wandb:
        with open("config/train_config.yaml") as f:
            train_cfg = yaml.safe_load(f)

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "pricing-bot"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"ppo-{args.config}-{args.timesteps // 1000}k",

            config={
                "policy_config": args.config,
                "total_timesteps": args.timesteps,
                "n_envs": args.n_envs,
                "seed": args.seed,

                # Sweep hyperparameters
                "learning_rate": args.learning_rate,
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
                "n_steps": args.n_steps,

                # Other training configuration
                **train_cfg,
            },
        )

    print("\n" + "=" * 50)
    print("Training PPO agent")
    print(f"Config      : {args.config}")
    print(f"Timesteps   : {args.timesteps:,}")
    print(f"W&B         : {'enabled' if use_wandb else 'disabled'}")
    print("=" * 50 + "\n")

    agent = PricingAgent(
        config_name=args.config,
        n_envs=args.n_envs,
        seed=args.seed,
        use_wandb=use_wandb,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
    )

    agent.train(total_timesteps=args.timesteps)

    agent.save(
        "outputs/models/ppo_best"
    )

    if use_wandb:
        wandb.finish()

    print("\nTraining complete.")
    print("Run `python evaluate.py` to compare against baselines.")


if __name__ == "__main__":
    main() 