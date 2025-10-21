import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import random
import shutil

# Suppress cosmetic matplotlib warnings
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

# Add parent directory to path before importing project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from breakout_dqn import BreakoutDQN
from core.CNN import CNN
from environment.reward_shaping import BreakoutRewardShaping
from helpers.FramePreprocess import preprocess_frame
from helpers.FrameStack import FrameStack

from experiments.experiment_config import (
    CHECKPOINT_FREQUENCY,
    EXPERIMENT_CONFIGS,
    FULL_TEST_EPISODES,
    GPU_OPTIMIZED_EPISODES,
    KEEP_BEST_N_CHECKPOINTS,
    MEDIUM_TEST_EPISODES,
    QUICK_TEST_EPISODES,
)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
warnings.filterwarnings("ignore", message="No artists with labels found")


class ExperimentRunner:
    def __init__(self, output_dir="experiments/results", use_gpu=True):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")

        # Create directories with error handling
        try:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        except OSError as e:
            print(f"❌ Error creating directories: {e}")
            print("💡 Tip: Check disk space with 'df -h'")
            raise

        self.results = {}

        # GPU setup
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if self.use_gpu:
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(
                f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  Running on CPU - experiments will be slower")

    def run_experiment(
        self, config_name, episodes=QUICK_TEST_EPISODES, render=False, resume_from=None
    ):
        """Run a single experiment with the given configuration."""
        if config_name not in EXPERIMENT_CONFIGS:
            raise ValueError(
                f"Unknown config: {config_name}. Available: {list(EXPERIMENT_CONFIGS.keys())}"
            )

        config = EXPERIMENT_CONFIGS[config_name]
        exp_checkpoint_dir = os.path.join(self.checkpoint_dir, config_name)

        try:
            os.makedirs(exp_checkpoint_dir, exist_ok=True)
        except OSError as e:
            print(f"❌ Cannot create checkpoint directory: {e}")
            raise

        print(f"\n{'=' * 80}")
        print(f"🧪 Running Experiment: {config['name']}")
        print(f"📝 Description: {config['description']}")
        print(f"📊 Episodes: {episodes}")
        print(f"💾 Checkpoints: {exp_checkpoint_dir}")
        print(f"🖥️  Device: {self.device}")
        if self.use_gpu:
            print(
                f"⚡ GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        print(f"{'=' * 80}\n")

        # Create DQN agent
        agent = BreakoutDQN()

        # Override reward shaping parameters
        agent.enable_reward_shaping = True
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
        start_episode = 0

        # Resume from checkpoint if specified
        if resume_from:
            print(f"📂 Resuming from checkpoint: {resume_from}")
            try:
                checkpoint = torch.load(resume_from, map_location=self.device)
                start_episode = checkpoint["episode"]
                print(f"   Starting from episode {start_episode}")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                print("   Starting from scratch...")
                resume_from = None

        # Run training
        try:
            rewards, epsilon_history, stats, checkpoints = self._run_training(
                agent,
                episodes,
                render,
                exp_checkpoint_dir,
                config_name,
                start_episode,
                resume_from,
            )

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
                "checkpoints": checkpoints,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "gpu_used": self.use_gpu,
            }

            # Save results
            self._save_experiment_results(config_name)

            # Print summary
            print(f"\n{'=' * 80}")
            print(
                f"✅ Experiment '{config_name}' completed in {duration / 3600:.2f} hours"
            )
            print("\n📊 COMPARISON METRICS (Original Game Score):")
            print(f"   Average Score: {metrics['avg_original_score']:.2f}")
            print(f"   Max Score: {metrics['max_original_score']:.2f}")
            print(f"   Final Average (last 10): {metrics['final_avg_original']:.2f}")
            print(f"   Improvement: {metrics['score_improvement']:.2f}")
            print("\n📈 LEARNING EFFICIENCY:")
            print(
                f"   Episodes to reach 5 score: {metrics.get('episodes_to_5', 'N/A')}"
            )
            print(
                f"   Episodes to reach 10 score: {metrics.get('episodes_to_10', 'N/A')}"
            )
            print(f"   Success rate (score > 0): {metrics['success_rate']:.1f}%")
            print("\n🎯 CONSISTENCY:")
            print(f"   Score std deviation: {metrics['std_original_score']:.2f}")
            print(f"   Median score: {metrics['median_original_score']:.2f}")
            print(f"   Top 10% average: {metrics['top_10_percent_avg']:.2f}")
            print("\n⏱️  SURVIVAL:")
            print(
                f"   Average episode length: {metrics['avg_episode_length']:.0f} steps"
            )
            print(f"   Max episode length: {metrics['max_episode_length']:.0f} steps")
            print(f"\n💾 Best checkpoint: {checkpoints.get('best', 'None')}")
            print(f"{'=' * 80}\n")

            # GPU memory summary
            if self.use_gpu:
                print(
                    f"🖥️  GPU Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                )
                torch.cuda.empty_cache()

            return self.results[config_name]

        except KeyboardInterrupt:
            print("\n⚠️  Experiment interrupted! Saving progress...")
            if config_name in self.results:
                self._save_experiment_results(config_name)
            raise
        except Exception as e:
            print(f"❌ Experiment '{config_name}' failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _run_training(
        self,
        agent,
        episodes,
        render,
        checkpoint_dir,
        exp_name,
        start_episode=0,
        resume_checkpoint=None,
    ):
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # GPU optimizations
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

        # Create environment
        base_env = gym.make(
            "ALE/Breakout-v5", render_mode="rgb_array" if render else None
        )

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

        # Load from checkpoint if resuming
        if resume_checkpoint:
            try:
                checkpoint = torch.load(
                    resume_checkpoint, map_location=agent.device, weights_only=False
                )
                policy_dqn.load_state_dict(checkpoint["policy_state_dict"])
                target_dqn.load_state_dict(checkpoint["target_state_dict"])
                agent.epsilon = checkpoint["epsilon"]
                # Note: Not loading memory to avoid serialization issues
                print(f"✅ Loaded checkpoint from episode {checkpoint['episode']}")
            except Exception as e:
                print(f"⚠️  Checkpoint loading failed: {e}, starting fresh")
                target_dqn.load_state_dict(policy_dqn.state_dict())
        else:
            target_dqn.load_state_dict(policy_dqn.state_dict())

        agent.optimizer = torch.optim.Adam(
            policy_dqn.parameters(), lr=agent.learning_rate
        )

        rewards_per_episode = []
        epsilon_history = []
        step_count = start_episode * 1000  # Approximate

        # Checkpoint tracking
        best_avg_original_score = -float("inf")
        saved_checkpoints = []

        # Statistics tracking
        episode_stats = {
            "lengths": [],
            "original_scores": [],
            "shaped_rewards": [],
            "paddle_hits": [],
            "blocks_broken": [],
            "side_bounces": [],
            "balls_lost": [],
            "losses": [],
            "lives_remaining": [],
        }

        # Training loop
        for episode in range(start_episode + 1, episodes + 1):
            obs, info = env.reset()
            lives = info.get("lives", 5)
            obs, _, terminated, truncated, info = env.step(1)
            obs = preprocess_frame(obs)
            frame_stack = FrameStack(4)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)

            state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
            total_shaped_reward = 0
            total_original_score = 0
            terminated = truncated = False
            episode_steps = 0
            episode_losses = []

            while not (terminated or truncated):
                # Epsilon-greedy action selection
                if random.random() < agent.epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_gpu = state.to(agent.device)
                        action = policy_dqn(state_gpu).argmax().item()

                next_obs, reward, terminated, truncated, info = env.step(action)

                # Track ORIGINAL game score
                if agent.enable_reward_shaping and "original_reward" in info:
                    total_original_score += info["original_reward"]
                else:
                    total_original_score += reward

                next_obs = preprocess_frame(next_obs)
                frame_stack.append(next_obs)
                next_state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                # Handle life loss
                current_lives = info.get("lives", lives)
                if current_lives < lives:
                    lives = current_lives
                    frame_stack.reset()
                    for _ in range(4):
                        frame_stack.append(next_obs)
                    next_state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                # Store transition
                agent.memory.append(
                    (state, action, next_state, reward, terminated or truncated)
                )
                state = next_state
                total_shaped_reward += reward
                episode_steps += 1
                step_count += 1

                # Training step
                if len(agent.memory) > agent.mini_batch_size and step_count % 4 == 0:
                    mini_batch = agent.memory.sample(agent.mini_batch_size)
                    loss = agent.optimize(mini_batch, policy_dqn, target_dqn)
                    episode_losses.append(loss)

                # Target network update
                if step_count % agent.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            # Episode complete - collect statistics
            rewards_per_episode.append(total_shaped_reward)
            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
            epsilon_history.append(agent.epsilon)

            # Store comprehensive statistics
            episode_stats["lengths"].append(episode_steps)
            episode_stats["original_scores"].append(total_original_score)
            episode_stats["shaped_rewards"].append(total_shaped_reward)
            episode_stats["lives_remaining"].append(current_lives)

            if len(episode_losses) > 0:
                episode_stats["losses"].append(np.mean(episode_losses))

            # Get EPISODE shaping stats
            if agent.enable_reward_shaping and hasattr(env, "get_shaping_stats"):
                ep_stats = env.get_shaping_stats()
                episode_stats["paddle_hits"].append(ep_stats["paddle_hits"])
                episode_stats["blocks_broken"].append(ep_stats["blocks_broken"])
                episode_stats["side_bounces"].append(ep_stats["side_bounces"])
                episode_stats["balls_lost"].append(ep_stats["balls_lost"])

            # Progress reporting
            if episode % 10 == 0:
                avg_shaped = np.mean(rewards_per_episode[-10:])
                avg_original = np.mean(episode_stats["original_scores"][-10:])

                print(
                    f"Episode {episode}/{episodes} | "
                    f"Original Score: {avg_original:.2f} | "
                    f"Shaped: {avg_shaped:.2f} | "
                    f"Epsilon: {agent.epsilon:.3f}"
                )

                if self.use_gpu and episode % 50 == 0:
                    print(
                        f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                    )

            # Checkpoint saving (ONLY network weights, not replay memory)
            if episode % CHECKPOINT_FREQUENCY == 0 or episode == episodes:
                avg_original_score = (
                    np.mean(episode_stats["original_scores"][-100:])
                    if len(episode_stats["original_scores"]) >= 100
                    else np.mean(episode_stats["original_scores"])
                )

                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_ep{episode:05d}_score{avg_original_score:.1f}.pt",
                )

                # Save ONLY essential data to avoid disk space issues
                checkpoint = {
                    "episode": episode,
                    "policy_state_dict": policy_dqn.state_dict(),
                    "target_state_dict": target_dqn.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "epsilon": agent.epsilon,
                    "avg_original_score": avg_original_score,
                    "step_count": step_count,
                    # Don't save: rewards, memory (too large)
                }

                try:
                    torch.save(checkpoint, checkpoint_path)
                    saved_checkpoints.append(
                        {
                            "path": checkpoint_path,
                            "episode": episode,
                            "avg_original_score": avg_original_score,
                        }
                    )
                    print(f"💾 Checkpoint saved: {checkpoint_path}")

                    # Track best checkpoint
                    if avg_original_score > best_avg_original_score:
                        best_avg_original_score = avg_original_score
                        best_checkpoint_path = os.path.join(
                            checkpoint_dir, "best_checkpoint.pt"
                        )
                        shutil.copy(checkpoint_path, best_checkpoint_path)
                        print(
                            f"🏆 New best checkpoint! Original score: {avg_original_score:.2f}"
                        )

                    # Cleanup old checkpoints
                    if len(saved_checkpoints) > KEEP_BEST_N_CHECKPOINTS + 1:
                        saved_checkpoints.sort(
                            key=lambda x: x["avg_original_score"], reverse=True
                        )
                        for old_ckpt in saved_checkpoints[
                            KEEP_BEST_N_CHECKPOINTS + 1 :
                        ]:
                            if (
                                os.path.exists(old_ckpt["path"])
                                and "best_checkpoint" not in old_ckpt["path"]
                            ):
                                try:
                                    os.remove(old_ckpt["path"])
                                    print(
                                        f"🗑️  Removed old checkpoint: {os.path.basename(old_ckpt['path'])}"
                                    )
                                except OSError:
                                    pass  # Ignore if can't delete
                        saved_checkpoints = saved_checkpoints[
                            : KEEP_BEST_N_CHECKPOINTS + 1
                        ]

                except OSError as e:
                    print(f"⚠️  Failed to save checkpoint: {e}")
                    print("💡 Tip: Check disk space with 'df -h'")

        env.close()

        # Checkpoint summary
        checkpoint_summary = {
            "total": len(saved_checkpoints),
            "best": (
                os.path.join(checkpoint_dir, "best_checkpoint.pt")
                if saved_checkpoints
                else None
            ),
            "final": saved_checkpoints[-1]["path"] if saved_checkpoints else None,
            "all": saved_checkpoints,
        }

        return rewards_per_episode, epsilon_history, episode_stats, checkpoint_summary

    def _calculate_metrics(self, shaped_rewards, stats):
        shaped_array = np.array(shaped_rewards)
        original_scores = np.array(stats["original_scores"])  # TRUE game scores

        metrics = {
            # === PRIMARY METRICS (Original Game Score) ===
            "avg_original_score": float(np.mean(original_scores)),
            "std_original_score": float(np.std(original_scores)),
            "max_original_score": float(np.max(original_scores)),
            "min_original_score": float(np.min(original_scores)),
            "median_original_score": float(np.median(original_scores)),
            # Learning progression
            "final_avg_original": (
                float(np.mean(original_scores[-10:]))
                if len(original_scores) >= 10
                else float(np.mean(original_scores))
            ),
            "initial_avg_original": (
                float(np.mean(original_scores[:10]))
                if len(original_scores) >= 10
                else float(np.mean(original_scores))
            ),
            "score_improvement": (
                float(np.mean(original_scores[-10:]) - np.mean(original_scores[:10]))
                if len(original_scores) >= 10
                else 0.0
            ),
            # Top performance
            "top_10_percent_avg": float(
                np.mean(
                    sorted(original_scores, reverse=True)[
                        : max(1, len(original_scores) // 10)
                    ]
                )
            ),
            "best_20_avg": float(
                np.mean(
                    sorted(original_scores, reverse=True)[
                        : min(20, len(original_scores))
                    ]
                )
            ),
            # === LEARNING EFFICIENCY ===
            "success_rate": float(
                100 * np.sum(original_scores > 0) / len(original_scores)
            ),
            "double_digit_rate": float(
                100 * np.sum(original_scores >= 10) / len(original_scores)
            ),
            # === SHAPED REWARD METRICS (for reference) ===
            "avg_shaped_reward": float(np.mean(shaped_array)),
            "max_shaped_reward": float(np.max(shaped_array)),
            # === CONSISTENCY METRICS ===
            "coefficient_of_variation": float(
                np.std(original_scores) / np.mean(original_scores)
                if np.mean(original_scores) > 0
                else float("inf")
            ),
        }

        # Learning speed metrics
        if len(original_scores) >= 10:
            # Episodes needed to reach milestones
            for threshold in [5, 10, 15, 20]:
                for i, score in enumerate(original_scores):
                    if score >= threshold:
                        metrics[f"episodes_to_{threshold}"] = i + 1
                        break
                else:
                    metrics[f"episodes_to_{threshold}"] = "Not reached"

        # Episode length metrics
        if "lengths" in stats and stats["lengths"]:
            lengths = np.array(stats["lengths"])
            metrics["avg_episode_length"] = float(np.mean(lengths))
            metrics["max_episode_length"] = float(np.max(lengths))
            metrics["median_episode_length"] = float(np.median(lengths))

            # Survival improvement
            if len(lengths) >= 10:
                metrics["length_improvement"] = float(
                    np.mean(lengths[-10:]) - np.mean(lengths[:10])
                )

        # Loss metrics
        if "losses" in stats and stats["losses"]:
            metrics["avg_loss"] = float(np.mean(stats["losses"]))
            metrics["final_avg_loss"] = (
                float(np.mean(stats["losses"][-10:]))
                if len(stats["losses"]) >= 10
                else float(np.mean(stats["losses"]))
            )

        # Lives remaining (better survival)
        if "lives_remaining" in stats and stats["lives_remaining"]:
            metrics["avg_lives_remaining"] = float(np.mean(stats["lives_remaining"]))

        # Shaping-specific metrics (for analysis, not comparison)
        if "paddle_hits" in stats and stats["paddle_hits"]:
            metrics["total_paddle_hits"] = int(np.sum(stats["paddle_hits"]))
            metrics["avg_paddle_hits"] = float(np.mean(stats["paddle_hits"]))
            metrics["total_blocks_broken"] = int(np.sum(stats["blocks_broken"]))
            metrics["avg_blocks_broken"] = float(np.mean(stats["blocks_broken"]))
            metrics["total_side_bounces"] = int(np.sum(stats["side_bounces"]))
            metrics["avg_side_bounces"] = float(np.mean(stats["side_bounces"]))
            metrics["total_balls_lost"] = int(np.sum(stats["balls_lost"]))
            metrics["avg_balls_lost"] = float(np.mean(stats["balls_lost"]))

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
            "device": result["device"],
            "gpu_used": result["gpu_used"],
            "checkpoints": {
                "total": result["checkpoints"]["total"],
                "best": result["checkpoints"]["best"],
                "final": result["checkpoints"]["final"],
            },
            "rewards": [float(r) for r in result["rewards"]],
            "epsilon_history": [float(e) for e in result["epsilon_history"]],
        }

        # Save to file
        filepath = os.path.join(self.output_dir, f"{config_name}_results.json")
        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"💾 Results saved to: {filepath}")

    def run_all_experiments(self, episodes=QUICK_TEST_EPISODES, render=False):
        """Run all configured experiments sequentially."""
        print(f"\n{'=' * 80}")
        print(f"🚀 Running ALL experiments ({len(EXPERIMENT_CONFIGS)} total)")
        print(f"📊 Episodes per experiment: {episodes}")
        print(f"🖥️  Device: {self.device}")
        print(f"{'=' * 80}\n")

        total_start = time.time()
        successful = 0
        failed = 0

        for i, config_name in enumerate(EXPERIMENT_CONFIGS.keys(), 1):
            print(f"\n{'=' * 80}")
            print(f"Progress: Experiment {i}/{len(EXPERIMENT_CONFIGS)}")
            print(f"{'=' * 80}")

            try:
                self.run_experiment(config_name, episodes, render)
                successful += 1

                # Clear GPU cache between experiments
                if self.use_gpu:
                    torch.cuda.empty_cache()

            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user. Saving progress...")
                break
            except Exception as e:
                print(f"❌ Experiment '{config_name}' failed: {e}")
                import traceback

                traceback.print_exc()
                print(f"⚠️  Skipping {config_name} due to error: {e}")
                failed += 1
                continue

        total_duration = time.time() - total_start

        print(f"\n{'=' * 80}")
        print("🏁 All experiments completed!")
        print(f"✅ Successful: {successful}/{len(EXPERIMENT_CONFIGS)}")
        print(f"❌ Failed: {failed}/{len(EXPERIMENT_CONFIGS)}")
        print(f"⏱️  Total time: {total_duration / 3600:.2f} hours")
        print(f"{'=' * 80}\n")

        # Generate comparison report
        if self.results:
            self.generate_comparison_report()

    def generate_comparison_report(self):
        if not self.results:
            print("⚠️  No results to compare")
            return

        print(f"\n{'=' * 80}")
        print("📊 EXPERIMENT COMPARISON REPORT")
        print(f"{'=' * 80}\n")

        # PRIMARY COMPARISON: Original Game Score
        print("🎮 PRIMARY METRIC: Original Game Score (Blocks Broken)")
        print(
            f"{'Experiment':<25} {'Avg Score':<12} {'Max Score':<12} {'Final Avg':<12} {'Success %':<12}"
        )
        print("-" * 90)

        sorted_by_score = sorted(
            self.results.items(),
            key=lambda x: x[1]["metrics"]["avg_original_score"],
            reverse=True,
        )

        for name, result in sorted_by_score:
            metrics = result["metrics"]
            print(
                f"{name:<25} {metrics['avg_original_score']:>11.2f} "
                f"{metrics['max_original_score']:>11.2f} "
                f"{metrics['final_avg_original']:>11.2f} "
                f"{metrics['success_rate']:>11.1f}%"
            )

        # LEARNING EFFICIENCY
        print("\n📈 LEARNING EFFICIENCY")
        print(
            f"{'Experiment':<25} {'To 5 pts':<12} {'To 10 pts':<12} {'Score Δ':<12} {'10+ Rate':<12}"
        )
        print("-" * 90)

        for name, result in sorted_by_score:
            metrics = result["metrics"]
            ep_5 = metrics.get("episodes_to_5", "N/A")
            ep_10 = metrics.get("episodes_to_10", "N/A")
            ep_5_str = f"{ep_5}" if isinstance(ep_5, int) else ep_5
            ep_10_str = f"{ep_10}" if isinstance(ep_10, int) else ep_10

            print(
                f"{name:<25} {ep_5_str:>11} {ep_10_str:>11} "
                f"{metrics['score_improvement']:>11.2f} "
                f"{metrics.get('double_digit_rate', 0):>11.1f}%"
            )

        # CONSISTENCY
        print("\n🎯 CONSISTENCY & RELIABILITY")
        print(
            f"{'Experiment':<25} {'Std Dev':<12} {'CoV':<12} {'Median':<12} {'Top 10%':<12}"
        )
        print("-" * 90)

        for name, result in sorted_by_score:
            metrics = result["metrics"]
            cov = metrics["coefficient_of_variation"]
            cov_str = f"{cov:.2f}" if cov != float("inf") else "inf"

            print(
                f"{name:<25} {metrics['std_original_score']:>11.2f} "
                f"{cov_str:>11} "
                f"{metrics['median_original_score']:>11.2f} "
                f"{metrics['top_10_percent_avg']:>11.2f}"
            )

        print("\n" + "=" * 90)

        # TOP PERFORMERS
        self._print_final_comparison(self.results)

        # Generate plots
        self.plot_comparison()

        # Save summary
        self._save_comparison_summary()

    def _print_final_comparison(self, all_results):
        print("\n" + "=" * 90)
        print()

        if not all_results:
            print("No results to compare")
            return

        # Sort by average original score
        sorted_by_score = sorted(
            all_results.items(),
            key=lambda x: x[1].get("metrics", {}).get("avg_original_score", 0),
            reverse=True,
        )

        # Sort by improvement
        sorted_by_improvement = sorted(
            all_results.items(),
            key=lambda x: x[1].get("metrics", {}).get("score_improvement", 0),
            reverse=True,
        )

        # Sort by consistency (lowest coefficient of variation)
        sorted_by_consistency = sorted(
            all_results.items(),
            key=lambda x: x[1]
            .get("metrics", {})
            .get("coefficient_of_variation", float("inf")),
        )

        print("TOP 3 by Average Original Score:")
        for i, (name, results) in enumerate(sorted_by_score[:3], 1):
            score = results["metrics"]["avg_original_score"]
            improvement = results["metrics"].get("score_improvement", 0)
            print(f"  {i}. {name}: {score:.2f} (Δ {improvement:+.2f})")

        print("\nFASTEST LEARNERS (Improvement):")
        for i, (name, results) in enumerate(sorted_by_improvement[:3], 1):
            improvement = results["metrics"].get("score_improvement", 0)
            final_score = results["metrics"].get("final_avg_original", 0)
            print(
                f"  {i}. {name}: {improvement:+.2f} improvement (final: {final_score:.2f})"
            )

        print("\nMOST CONSISTENT (Lowest CoV):")
        for i, (name, results) in enumerate(sorted_by_consistency[:3], 1):
            cov = results["metrics"]["coefficient_of_variation"]
            avg_score = results["metrics"]["avg_original_score"]
            print(f"  {i}. {name}: CoV={cov:.2f} (avg: {avg_score:.2f})")

        print()
        print("=" * 90)

    def _save_comparison_summary(self):
        summary = {
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "gpu_used": self.use_gpu,
            "num_experiments": len(self.results),
            "rankings": {
                "by_avg_original_score": sorted(
                    [
                        (name, result["metrics"]["avg_original_score"])
                        for name, result in self.results.items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                ),
                "by_score_improvement": sorted(
                    [
                        (name, result["metrics"]["score_improvement"])
                        for name, result in self.results.items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                ),
                "by_max_original_score": sorted(
                    [
                        (name, result["metrics"]["max_original_score"])
                        for name, result in self.results.items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                ),
                "by_success_rate": sorted(
                    [
                        (name, result["metrics"]["success_rate"])
                        for name, result in self.results.items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                ),
                "by_consistency": sorted(
                    [
                        (name, result["metrics"]["coefficient_of_variation"])
                        for name, result in self.results.items()
                        if result["metrics"]["coefficient_of_variation"] != float("inf")
                    ],
                    key=lambda x: x[1],
                    reverse=False,  # Lower is better
                ),
            },
            "experiments": {
                name: result["metrics"] for name, result in self.results.items()
            },
        }

        filepath = os.path.join(self.output_dir, "comparison_summary.json")
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Comparison summary saved to: {filepath}")

    def _get_config_color(self, config_name):
        """Get consistent color for each configuration."""
        color_map = {
            "baseline": "#1f77b4",
            "paddle_hit_only": "#ff7f0e",
            "center_position_only": "#2ca02c",
            "side_angle_only": "#d62728",
            "block_multiplier_only": "#9467bd",
            "ball_loss_penalty_only": "#8c564b",
            "survival_bonus_only": "#e377c2",
            "all_combined": "#7f7f7f",
            "bonuses_only": "#17becf",
        }
        return color_map.get(config_name, "#cccccc")

    def plot_comparison(self):
        if not self.results:
            return

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

        fig.suptitle(
            "Reward Shaping Experiments - Performance Comparison",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        # Get sorted results
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]["metrics"]["avg_original_score"],
            reverse=True,
        )
        names = [name for name, _ in sorted_results]

        # ============================================================
        # PLOT 1: Learning Curves
        # ============================================================
        ax_learning = fig.add_subplot(gs[:, :2])

        # Group experiments by performance tier for styling
        top_performers = set(names[:3])
        mid_performers = set(names[3:6]) if len(names) > 3 else set()

        has_enough_data = False

        # Plot with different styles for clarity
        for name in names:
            result = self.results[name]
            scores = result["stats"]["original_scores"]

            # Adaptive smoothing based on data size
            if len(scores) >= 50:
                window = max(20, len(scores) // 50)
                moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
                x_data = range(len(moving_avg))
                marker = None
                markersize = 0
                has_enough_data = True
            elif len(scores) >= 10:
                window = min(5, len(scores) // 2)
                moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
                x_data = range(len(moving_avg))
                marker = "o"
                markersize = 6
            else:
                moving_avg = scores
                x_data = range(len(scores))
                marker = "o"
                markersize = 8

            if len(moving_avg) == 0:
                continue

            # Style based on performance tier
            if name in top_performers:
                linewidth = 3.5 if has_enough_data else 2.5
                alpha = 0.95
                linestyle = "-"
                zorder = 10
            elif name in mid_performers:
                linewidth = 2.5 if has_enough_data else 2.0
                alpha = 0.75
                linestyle = "--"
                zorder = 5
            else:
                # Low performers
                linewidth = 1.8 if has_enough_data else 1.5
                alpha = 0.55
                linestyle = ":"
                zorder = 1

            # Special handling for baseline - always prominent
            if name == "baseline":
                linewidth = 4.0 if has_enough_data else 3.0
                linestyle = "-"
                alpha = 1.0
                zorder = 15

            # Plot the line
            ax_learning.plot(
                x_data,
                moving_avg,
                label=name.replace("_", " ").title(),
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
                color=self._get_config_color(name),
                marker=marker,
                markersize=markersize,
                markevery=max(1, len(x_data) // 10) if marker else None,
                zorder=zorder,
            )

        ax_learning.set_xlabel("Episode", fontsize=14, fontweight="bold")
        ax_learning.set_ylabel(
            "Original Game Score (Moving Avg)", fontsize=14, fontweight="bold"
        )

        ax_learning.set_title(
            "Learning Progress Over Time",
            fontsize=16,
            fontweight="bold",
            pad=25,
        )

        ax_learning.legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),  # Position inside but at top-left corner
            fontsize=8,
            framealpha=0.95,
            edgecolor="black",
            title="Configuration (by performance)",
            title_fontsize=9,
            ncol=2 if len(names) > 6 else 1,
            columnspacing=0.8,
            labelspacing=0.6,
            borderpad=0.5,
            handlelength=1.5,
            handleheight=0.7,
        )

        ax_learning.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax_learning.set_ylim(bottom=0)

        # ============================================================
        # PLOT 2: Final Performance Ranking
        # ============================================================
        ax_ranking = fig.add_subplot(gs[0, 2])

        avg_scores = [
            self.results[name]["metrics"]["avg_original_score"] for name in names
        ]
        colors = [self._get_config_color(name) for name in names]

        ax_ranking.barh(
            range(len(names)),
            avg_scores,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        ax_ranking.set_yticks(range(len(names)))
        ax_ranking.set_yticklabels(
            [n.replace("_", " ").title() for n in names], fontsize=8
        )
        ax_ranking.set_xlabel("Average Score", fontsize=12, fontweight="bold")
        ax_ranking.set_title(
            "Overall Performance", fontsize=14, fontweight="bold", pad=15
        )
        ax_ranking.grid(axis="x", alpha=0.3, linestyle="--")

        # Add value labels
        max_score = max(avg_scores)
        for i, val in enumerate(avg_scores):
            # Position label to the right of bar
            label_x = val + (max_score * 0.03)
            ax_ranking.text(
                label_x,
                i,
                f"{val:.1f}",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        ax_ranking.set_xlim(right=max_score * 1.2)

        # ============================================================
        # PLOT 3: Learning Efficiency
        # ============================================================
        ax_efficiency = fig.add_subplot(gs[1, 2])

        improvements = [
            self.results[name]["metrics"]["score_improvement"] for name in names
        ]

        # Color by improvement direction
        colors_eff = [
            (
                "#2ecc71"
                if x > 2
                else "#27ae60" if x > 0 else "#e74c3c" if x < -2 else "#95a5a6"
            )
            for x in improvements
        ]

        ax_efficiency.barh(
            range(len(names)),
            improvements,
            color=colors_eff,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        ax_efficiency.axvline(x=0, color="black", linestyle="--", linewidth=2)
        ax_efficiency.set_yticks(range(len(names)))
        ax_efficiency.set_yticklabels(
            [n.replace("_", " ").title() for n in names], fontsize=8
        )
        ax_efficiency.set_xlabel("Score Improvement", fontsize=12, fontweight="bold")
        ax_efficiency.set_title(
            "Learning Efficiency (Δ)", fontsize=14, fontweight="bold", pad=15
        )
        ax_efficiency.grid(axis="x", alpha=0.3, linestyle="--")

        # Add value labels with smart positioning
        if improvements:  # Safety check
            max_abs_val = max(abs(min(improvements)), abs(max(improvements)))
            for i, val in enumerate(improvements):
                if abs(val) > 0.3:  # Only show if significant
                    # Position based on value direction
                    offset = max(max_abs_val * 0.08, 0.1)  # At least 0.1 offset
                    if val > 0:
                        x_pos = val + offset
                        ha = "left"
                    else:
                        x_pos = val - offset
                        ha = "right"

                    ax_efficiency.text(
                        x_pos,
                        i,
                        f"{val:+.1f}",
                        va="center",
                        ha=ha,
                        fontsize=8,
                        fontweight="bold",
                    )

            # Adjust x-axis limits for label visibility
            x_margin = max(max_abs_val * 0.25, 0.5)  # At least 0.5 margin
            ax_efficiency.set_xlim(
                left=min(improvements) - x_margin, right=max(improvements) + x_margin
            )

        # Add summary text box at bottom
        summary_text = (
            f"Total Experiments: {len(self.results)} | "
            f"Best Config: {names[0].replace('_', ' ').title()} ({avg_scores[0]:.1f}) | "
            f"Device: {self.device}"
        )

        fig.text(
            0.5,
            0.015,
            summary_text,
            fontsize=10,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5, pad=0.6),
        )

        plt.tight_layout(rect=[0, 0.04, 1, 0.97])

        plot_path = os.path.join(self.output_dir, "experiment_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
        print(f"\n📈 Comparison plots saved to: {plot_path}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run reward shaping experiments with GPU support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (100 episodes)
  python experiments/experiment_runner.py --config baseline --episodes 100
  
  # GPU-optimized long run (5000 episodes)
  python experiments/experiment_runner.py --config all_combined --episodes 5000 --gpu
  
  # Run all experiments
  python experiments/experiment_runner.py --config all --episodes 500
  
  # Resume from checkpoint
  python experiments/experiment_runner.py --config baseline --resume experiments/results/checkpoints/baseline/checkpoint_ep00500_reward15.3.pt
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="all",
        help='Specific config to run or "all" for all experiments',
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=QUICK_TEST_EPISODES,
        help=f"Number of episodes (default: {QUICK_TEST_EPISODES}, GPU recommended: {GPU_OPTIMIZED_EPISODES})",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during training (slows down significantly)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available configurations"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU if available (default: True)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint file"
    )

    args = parser.parse_args()

    # List configurations
    if args.list:
        print("\n📋 Available Experiment Configurations:\n")
        for name, config in EXPERIMENT_CONFIGS.items():
            print(f"  • {name:<25} - {config['description']}")
        print("\n💡 Recommended episodes:")
        print(f"  • Quick test: {QUICK_TEST_EPISODES}")
        print(f"  • Medium test: {MEDIUM_TEST_EPISODES}")
        print(f"  • Full test: {FULL_TEST_EPISODES}")
        print(f"  • GPU-optimized: {GPU_OPTIMIZED_EPISODES}")
        print()
        return

    # Determine GPU usage
    use_gpu = args.gpu and not args.cpu

    # Run experiments
    runner = ExperimentRunner(use_gpu=use_gpu)

    if args.config == "all":
        runner.run_all_experiments(episodes=args.episodes, render=args.render)
    else:
        if args.config not in EXPERIMENT_CONFIGS:
            print(f"❌ Unknown configuration: {args.config}")
            print(f"Available: {', '.join(EXPERIMENT_CONFIGS.keys())}")
            return

        runner.run_experiment(
            args.config,
            episodes=args.episodes,
            render=args.render,
            resume_from=args.resume,
        )


if __name__ == "__main__":
    main()
