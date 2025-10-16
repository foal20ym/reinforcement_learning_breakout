"""
DQN Agent for Atari Breakout - Main Entry Point

This script provides options to:
1. Train a new DQN agent
2. Evaluate a trained agent
3. Watch random gameplay (for testing environment setup)

Usage:
    # Train agent
    python main.py --train

    # Evaluate agent
    python main.py --evaluate --checkpoint checkpoints/dqn_breakout_final.pt

    # Watch random gameplay
    python main.py --random
"""

import argparse
import gymnasium as gym
import time
import ale_py

gym.register_envs(ale_py)


def random_play(n_episodes=3, max_steps=10000):
    """Play random actions for testing."""
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    try:
        for ep in range(1, n_episodes + 1):
            obs, info = env.reset()
            terminated = truncated = False
            total_reward = 0.0
            step = 0
            while not (terminated or truncated) and step < max_steps:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                time.sleep(1 / 60)
            print(f"Episode {ep} finished in {step} steps, total reward: {total_reward}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="DQN Breakout Agent")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint (path or 'latest')")
    parser.add_argument("--fresh", action="store_true", help="Start fresh training (ignore existing checkpoints)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the agent")
    parser.add_argument("--random", action="store_true", help="Watch random gameplay")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/dqn_breakout_final.pt", help="Checkpoint path for evaluation"
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for evaluation")

    args = parser.parse_args()

    if args.train or args.resume or args.fresh:
        print("Starting training...")
        from train import train

        if args.fresh:
            train(resume_from=None)
        elif args.resume:
            train(resume_from=args.resume)
        else:
            # Let train.py handle auto-detection
            import sys

            sys.argv = ["train.py"]  # Clear args for train.py
            exec(open("train.py").read())
    elif args.evaluate:
        print("Starting evaluation...")
        from evaluate import evaluate

        evaluate(args.checkpoint, args.episodes, render=True)
    elif args.random:
        print("Starting random gameplay...")
        random_play()
    else:
        print("Please specify --train, --evaluate, or --random")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
