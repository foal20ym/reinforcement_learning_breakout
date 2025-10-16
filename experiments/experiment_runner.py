"""
Experiment runner for testing different reward shaping configurations.
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakout_dqn import BreakoutDQN
from experiments.experiment_config import EXPERIMENT_CONFIGS, QUICK_TEST_EPISODES, FULL_TEST_EPISODES


class ExperimentRunner:
    def __init__(self, output_dir="experiments/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}

    def run_experiment(self, config_name, episodes=QUICK_TEST_EPISODES, render=False):
        """Run a single experiment with the given configuration."""
        if config_name not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(EXPERIMENT_CONFIGS.keys())}")

        config = EXPERIMENT_CONFIGS[config_name]
        print(f"\n{'='*80}")
        print(f"🧪 Running Experiment: {config['name']}")
        print(f"📝 Description: {config['description']}")
        print(f"📊 Episodes: {episodes}")
        print(f"{'='*80}\n")

        # Create DQN agent
        agent = BreakoutDQN()

        # Override reward shaping parameters
        agent.enable_reward_shaping = True  # Always enable wrapper for fair comparison
        agent.reward_shaping_params = {
            "paddle_hit_bonus": config["paddle_hit_bonus"],
            "center_position_bonus": config["center_position_bonus"],
            "side_angle_bonus": config["side_angle_bonus"],
            "block_bonus_multiplier": config["block_bonus_multiplier"],
            "ball_loss_penalty": config["ball_loss_penalty"],
            "survival_bonus": config.get("survival_bonus", 0.0),
        }

        # For baseline, disable shaping entirely
        if config_name == "baseline":
            agent.enable_reward_shaping = False

        # Track start time
        start_time = time.time()

        # Run training
        try:
            rewards, epsilon_history, stats = self._run_training(agent, episodes, render)

            # Calculate metrics
            duration = time.time() - start_time
            metrics = self._calculate_metrics(rewards, stats)

            # Store results
            self.results[config_name] = {
                "config": config,
                "episodes": episodes,
                "rewards": rewards,
                "epsilon_history": epsilon_history,
                "stats": stats,
                "metrics": metrics,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
            }

            # Save results
            self._save_experiment_results(config_name)

            print(f"\n✅ Experiment '{config_name}' completed in {duration:.2f}s")
            print(f"📊 Average reward: {metrics['avg_reward']:.2f}")
            print(f"🎯 Max reward: {metrics['max_reward']:.2f}")
            print(f"📈 Final 10-episode average: {metrics['final_avg']:.2f}\n")

            return self.results[config_name]

        except Exception as e:
            print(f"❌ Experiment '{config_name}' failed: {e}")
            raise

    def _run_training(self, agent, episodes, render):
        """Modified training loop that captures rewards and stats."""
        from visualization import plot_progress
        from core.ReplayMemory import ReplayMemory
        from helpers.FramePreprocess import preprocess_frame
        from helpers.FrameStack import FrameStack
        from core.CNN import CNN
        from environment.reward_shaping import BreakoutRewardShaping
        import torch
        import gymnasium as gym
        import numpy as np
        import random

        # Create environment
        base_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array" if render else None)

        if agent.enable_reward_shaping:
            env = BreakoutRewardShaping(
                base_env,
                **agent.reward_shaping_params,
                enable_shaping=True,
            )
        else:
            env = base_env

        num_actions = env.action_space.n

        # Initialize networks
        policy_dqn = CNN(num_actions).to(agent.device)
        target_dqn = CNN(num_actions).to(agent.device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        agent.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=agent.learning_rate)

        rewards_per_episode = []
        epsilon_history = []
        step_count = 0

        # Track additional statistics
        episode_stats = {
            "lengths": [],
            "original_rewards": [],
            "shaped_rewards": [],
            "paddle_hits": [],
            "blocks_broken": [],
            "side_bounces": [],
            "balls_lost": [],
        }

        for episode in range(1, episodes + 1):
            obs, info = env.reset()
            lives = info.get("lives", 5)
            obs, _, terminated, truncated, info = env.step(1)
            obs = preprocess_frame(obs)
            frame_stack = FrameStack(4)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)

            state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
            total_reward = 0
            total_original_reward = 0
            terminated = truncated = False
            episode_steps = 0

            while not (terminated or truncated):
                if random.random() < agent.epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.to(agent.device)).argmax().item()

                next_obs, reward, terminated, truncated, info = env.step(action)

                if agent.enable_reward_shaping and "original_reward" in info:
                    total_original_reward += info["original_reward"]
                else:
                    total_original_reward += reward

                next_obs = preprocess_frame(next_obs)
                frame_stack.append(next_obs)
                next_state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                current_lives = info.get("lives", lives)
                if current_lives < lives:
                    lives = current_lives
                    frame_stack.reset()
                    for _ in range(4):
                        frame_stack.append(next_obs)
                    next_state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                agent.memory.append((state, action, next_state, reward, terminated or truncated))
                state = next_state
                total_reward += reward
                episode_steps += 1
                step_count += 1

                if len(agent.memory) > agent.mini_batch_size and step_count % 4 == 0:
                    mini_batch = agent.memory.sample(agent.mini_batch_size)
                    agent.optimize(mini_batch, policy_dqn, target_dqn)

                if step_count % agent.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode.append(total_reward)
            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
            epsilon_history.append(agent.epsilon)

            # Collect episode statistics
            episode_stats["lengths"].append(episode_steps)
            episode_stats["original_rewards"].append(total_original_reward)
            episode_stats["shaped_rewards"].append(total_reward)

            if agent.enable_reward_shaping and hasattr(env, "get_shaping_stats"):
                stats = env.get_shaping_stats()
                episode_stats["paddle_hits"].append(stats["paddle_hits"])
                episode_stats["blocks_broken"].append(stats["blocks_broken"])
                episode_stats["side_bounces"].append(stats["side_bounces"])
                episode_stats["balls_lost"].append(stats["balls_lost"])

            # Print progress every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_per_episode[-10:])
                print(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        env.close()

        return rewards_per_episode, epsilon_history, episode_stats

    def _calculate_metrics(self, rewards, stats):
        """Calculate performance metrics from experiment results."""
        rewards_array = np.array(rewards)

        metrics = {
            "avg_reward": float(np.mean(rewards_array)),
            "std_reward": float(np.std(rewards_array)),
            "max_reward": float(np.max(rewards_array)),
            "min_reward": float(np.min(rewards_array)),
            "median_reward": float(np.median(rewards_array)),
            "final_avg": float(np.mean(rewards_array[-10:])),  # Last 10 episodes
            "initial_avg": float(np.mean(rewards_array[:10])),  # First 10 episodes
            "improvement": float(np.mean(rewards_array[-10:]) - np.mean(rewards_array[:10])),
        }

        # Add episode length metrics
        if "lengths" in stats:
            metrics["avg_episode_length"] = float(np.mean(stats["lengths"]))

        # Add shaping-specific metrics
        if "paddle_hits" in stats:
            metrics["total_paddle_hits"] = int(np.sum(stats["paddle_hits"]))
            metrics["total_blocks_broken"] = int(np.sum(stats["blocks_broken"]))
            metrics["total_side_bounces"] = int(np.sum(stats["side_bounces"]))
            metrics["total_balls_lost"] = int(np.sum(stats["balls_lost"]))

        return metrics

    def _save_experiment_results(self, config_name):
        """Save experiment results to JSON file."""
        result = self.results[config_name]

        # Prepare data for JSON serialization
        json_data = {
            "config": result["config"],
            "episodes": result["episodes"],
            "metrics": result["metrics"],
            "duration": result["duration"],
            "timestamp": result["timestamp"],
            "rewards": [float(r) for r in result["rewards"]],
            "epsilon_history": [float(e) for e in result["epsilon_history"]],
        }

        # Save to file
        filepath = os.path.join(self.output_dir, f"{config_name}_results.json")
        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"💾 Results saved to: {filepath}")

    def run_all_experiments(self, episodes=QUICK_TEST_EPISODES, render=False):
        """Run all configured experiments."""
        print(f"\n{'='*80}")
        print(f"🚀 Running ALL experiments ({len(EXPERIMENT_CONFIGS)} total)")
        print(f"📊 Episodes per experiment: {episodes}")
        print(f"{'='*80}\n")

        for config_name in EXPERIMENT_CONFIGS.keys():
            try:
                self.run_experiment(config_name, episodes, render)
            except Exception as e:
                print(f"⚠️  Skipping {config_name} due to error: {e}")
                continue

        # Generate comparison report
        self.generate_comparison_report()

    def generate_comparison_report(self):
        """Generate a comparison report of all experiments."""
        if not self.results:
            print("⚠️  No results to compare")
            return

        print(f"\n{'='*80}")
        print("📊 EXPERIMENT COMPARISON REPORT")
        print(f"{'='*80}\n")

        # Create comparison table
        print(f"{'Experiment':<25} {'Avg Reward':<12} {'Max Reward':<12} {'Improvement':<12}")
        print("-" * 80)

        for name, result in sorted(self.results.items(), key=lambda x: x[1]["metrics"]["avg_reward"], reverse=True):
            metrics = result["metrics"]
            print(
                f"{name:<25} {metrics['avg_reward']:>11.2f} {metrics['max_reward']:>11.2f} {metrics['improvement']:>11.2f}"
            )

        print("\n" + "=" * 80)

        # Generate plots
        self.plot_comparison()

    def plot_comparison(self):
        """Generate comparison plots."""
        if not self.results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Reward Shaping Experiments Comparison", fontsize=16, fontweight="bold")

        # Plot 1: Average rewards
        ax1 = axes[0, 0]
        names = []
        avg_rewards = []
        colors = []

        for name, result in sorted(self.results.items(), key=lambda x: x[1]["metrics"]["avg_reward"], reverse=True):
            names.append(name)
            avg_rewards.append(result["metrics"]["avg_reward"])
            colors.append("green" if name == "all_combined" else "blue" if name == "baseline" else "orange")

        bars = ax1.barh(names, avg_rewards, color=colors, alpha=0.7)
        ax1.set_xlabel("Average Reward")
        ax1.set_title("Average Reward by Configuration")
        ax1.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, avg_rewards)):
            ax1.text(val, i, f" {val:.1f}", va="center")

        # Plot 2: Improvement over baseline
        ax2 = axes[0, 1]
        baseline_reward = self.results.get("baseline", {}).get("metrics", {}).get("avg_reward", 0)
        improvements = []

        for name in names:
            if name == "baseline":
                improvements.append(0)
            else:
                improvement = self.results[name]["metrics"]["avg_reward"] - baseline_reward
                improvements.append(improvement)

        colors2 = ["red" if x < 0 else "green" for x in improvements]
        ax2.barh(names, improvements, color=colors2, alpha=0.7)
        ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Improvement over Baseline")
        ax2.set_title("Reward Improvement vs Baseline")
        ax2.grid(axis="x", alpha=0.3)

        # Plot 3: Learning curves
        ax3 = axes[1, 0]
        for name, result in self.results.items():
            rewards = result["rewards"]
            # Plot moving average
            window = min(10, len(rewards) // 5)
            if window > 0:
                moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ax3.plot(moving_avg, label=name, alpha=0.7)

        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Reward (Moving Average)")
        ax3.set_title("Learning Curves")
        ax3.legend(loc="best", fontsize=8)
        ax3.grid(alpha=0.3)

        # Plot 4: Shaping statistics (if available)
        ax4 = axes[1, 1]
        stat_types = ["paddle_hits", "blocks_broken", "side_bounces", "balls_lost"]
        stat_names = ["Paddle Hits", "Blocks Broken", "Side Bounces", "Balls Lost"]

        # Collect data
        exp_names = []
        stat_data = {stat: [] for stat in stat_types}

        for name, result in self.results.items():
            metrics = result["metrics"]
            if any(f"total_{stat}" in metrics for stat in stat_types):
                exp_names.append(name)
                for stat in stat_types:
                    key = f"total_{stat}"
                    stat_data[stat].append(metrics.get(key, 0))

        if exp_names:
            x = np.arange(len(exp_names))
            width = 0.2

            for i, (stat, label) in enumerate(zip(stat_types, stat_names)):
                offset = width * (i - 1.5)
                ax4.bar(x + offset, stat_data[stat], width, label=label, alpha=0.7)

            ax4.set_xlabel("Experiment")
            ax4.set_ylabel("Count")
            ax4.set_title("Shaping Event Statistics")
            ax4.set_xticks(x)
            ax4.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=8)
            ax4.legend(fontsize=8)
            ax4.grid(axis="y", alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No shaping statistics available", ha="center", va="center", transform=ax4.transAxes)
            ax4.set_title("Shaping Event Statistics")

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.output_dir, "experiment_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\n📈 Comparison plots saved to: {plot_path}")

        plt.close()


def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run reward shaping experiments")
    parser.add_argument("--config", type=str, help='Specific config to run (or "all")')
    parser.add_argument(
        "--episodes", type=int, default=QUICK_TEST_EPISODES, help=f"Number of episodes (default: {QUICK_TEST_EPISODES})"
    )
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--list", action="store_true", help="List available configurations")

    args = parser.parse_args()

    # List configurations
    if args.list:
        print("\n📋 Available Experiment Configurations:\n")
        for name, config in EXPERIMENT_CONFIGS.items():
            print(f"  • {name:<25} - {config['description']}")
        print()
        return

    # Run experiments
    runner = ExperimentRunner()

    if args.config == "all" or args.config is None:
        runner.run_all_experiments(episodes=args.episodes, render=args.render)
    else:
        if args.config not in EXPERIMENT_CONFIGS:
            print(f"❌ Unknown configuration: {args.config}")
            print(f"Available: {', '.join(EXPERIMENT_CONFIGS.keys())}")
            return

        runner.run_experiment(args.config, episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()
