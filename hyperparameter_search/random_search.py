import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Add parent directory to path to import project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from breakout_dqn import BreakoutDQN
from core.CNN import CNN
from helpers.FramePreprocess import preprocess_frame
from helpers.FrameStack import FrameStack


class RandomHyperparameterSearch:
    """
    Random hyperparameter search for DQN baseline (no reward shaping).

    Searches the most impactful parameters:
    - learning_rate: Controls optimization speed
    - epsilon_decay: Balances exploration vs exploitation
    - mini_batch_size: Training batch size

    Optimized for speed: short episodes, GPU support, early stopping.
    """

    def __init__(self, output_dir="hyperparameter_search/results", use_gpu=True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

        # GPU setup
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if self.use_gpu:
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True  # Speed optimization
        else:
            print("⚙️  Running on CPU")

    def search(self, param_ranges, n_trials=5, test_episodes=50):
        """
        Test N random parameter combinations.

        Args:
            param_ranges: Dict of parameter ranges to search
            n_trials: Number of random configs to test (5-10 recommended)
            test_episodes: Episodes per test (20-30 for speed, 50+ for accuracy)
        """
        print(f"\n{'=' * 70}")
        print("🎲 RANDOM HYPERPARAMETER SEARCH")
        print(f"Device: {self.device}")
        print(f"Trials: {n_trials}")
        print(f"Episodes per trial: {test_episodes}")
        print(f"{'=' * 70}\n")

        for trial in range(1, n_trials + 1):
            # Sample random parameters
            params = self._sample_params(param_ranges)

            print(f"\n{'─' * 70}")
            print(f"Trial {trial}/{n_trials}")
            for key, val in params.items():
                print(
                    f"  {key}: {val:.5f}"
                    if isinstance(val, float)
                    else f"  {key}: {val}"
                )
            print(f"{'─' * 70}")

            # Run quick test
            score = self._test_config(params, test_episodes)

            # Save result
            self.results.append(
                {
                    "trial": trial,
                    "params": params,
                    "score": score,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            self._save_results()
            print(f"✅ Trial {trial} Score: {score:.2f}\n")

            # Clear GPU cache
            if self.use_gpu:
                torch.cuda.empty_cache()

        self._print_summary()
        return self.results

    def _sample_params(self, param_ranges):
        """Sample random parameters from ranges."""
        params = {}
        for name, values in param_ranges.items():
            if isinstance(values, tuple):  # Continuous range (min, max)
                params[name] = random.uniform(*values)
            else:  # Discrete choices
                params[name] = random.choice(values)
        return params

    def _test_config(self, params, episodes):
        """
        Run a fast training test with given parameters.

        Optimizations:
        - Short episodes (20-30)
        - GPU acceleration
        - No checkpointing
        - Simplified tracking
        """
        try:
            # Setup agent with custom params
            agent = BreakoutDQN()
            agent.learning_rate = float(
                params.get("learning_rate", agent.learning_rate)
            )
            agent.epsilon_decay = float(
                params.get("epsilon_decay", agent.epsilon_decay)
            )
            agent.mini_batch_size = int(
                params.get("mini_batch_size", agent.mini_batch_size)
            )
            agent.device = self.device  # Use specified device

            # Disable reward shaping for baseline
            agent.enable_reward_shaping = False

            # Setup environment
            env = gym.make("ALE/Breakout-v5", render_mode=None)
            num_actions = env.action_space.n

            # Networks on GPU/CPU
            policy_dqn = CNN(num_actions).to(self.device)
            target_dqn = CNN(num_actions).to(self.device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            agent.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=agent.learning_rate
            )

            scores = []
            step_count = 0

            # Training loop
            for ep in range(1, episodes + 1):
                obs, info = env.reset()
                obs, _, _, _, _ = env.step(1)  # Fire to start
                obs = preprocess_frame(obs)

                frame_stack = FrameStack(4)
                frame_stack.reset()
                for _ in range(4):
                    frame_stack.append(obs)

                state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
                episode_score = 0
                done = False

                while not done:
                    # Epsilon-greedy action selection
                    if random.random() < agent.epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            state_gpu = state.to(self.device)
                            action = policy_dqn(state_gpu).argmax().item()

                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    next_obs = preprocess_frame(next_obs)
                    frame_stack.append(next_obs)
                    next_state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                    # Store transition
                    agent.memory.append((state, action, next_state, reward, done))
                    state = next_state
                    episode_score += reward
                    step_count += 1

                    # Training step (every 4 steps)
                    if (
                        len(agent.memory) > agent.mini_batch_size
                        and step_count % 4 == 0
                    ):
                        mini_batch = agent.memory.sample(agent.mini_batch_size)
                        agent.optimize(mini_batch, policy_dqn, target_dqn)

                    # Sync target network
                    if step_count % agent.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

                scores.append(episode_score)
                agent.epsilon = max(
                    agent.epsilon * agent.epsilon_decay, agent.epsilon_min
                )

                # Progress update every 10 episodes
                if ep % 10 == 0:
                    recent_avg = (
                        np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
                    )
                    print(f"  Episode {ep}/{episodes}: Recent Avg = {recent_avg:.1f}")

            env.close()

            # Return average of last 10 episodes (most stable)
            final_score = (
                float(np.mean(scores[-10:]))
                if len(scores) >= 10
                else float(np.mean(scores))
            )
            return final_score

        except Exception as e:
            print(f"⚠️  Trial failed: {e}")
            import traceback

            traceback.print_exc()
            return 0.0

    def _save_results(self):
        """Save results after each trial (crash recovery)."""
        filepath = os.path.join(self.output_dir, "random_search_results.json")
        with open(filepath, "w") as f:
            json.dump(
                {
                    "results": self.results,
                    "num_trials": len(self.results),
                    "device": str(self.device),
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def _print_summary(self):
        """Print top configurations ranked by score."""
        if not self.results:
            print("⚠️  No results to summarize")
            return

        sorted_results = sorted(self.results, key=lambda x: x["score"], reverse=True)

        print(f"\n{'=' * 70}")
        print("🏆 TOP 3 CONFIGURATIONS (Ranked by Score)")
        print(f"{'=' * 70}\n")

        for i, result in enumerate(sorted_results[:3], 1):
            print(f"{i}. Score: {result['score']:.2f}")
            print(f"   learning_rate: {result['params']['learning_rate']:.6f}")
            print(f"   epsilon_decay: {result['params']['epsilon_decay']:.4f}")
            print(f"   mini_batch_size: {result['params']['mini_batch_size']}")
            print()

        # Save best config separately
        best_config_path = os.path.join(self.output_dir, "best_config.json")
        with open(best_config_path, "w") as f:
            json.dump(sorted_results[0], f, indent=2)

        results_path = os.path.join(self.output_dir, "random_search_results.json")
        print(f"💾 All results: {results_path}")
        print(f"💾 Best config: {best_config_path}")
        print(f"{'=' * 70}\n")


def main():
    searcher = RandomHyperparameterSearch(use_gpu=True)

    # Define search space
    param_ranges = {
        "learning_rate": (3e-5, 5e-4),
        "epsilon_decay": [0.985, 0.990, 0.993, 0.995, 0.997, 0.998],
        "mini_batch_size": [32, 64, 128],
    }

    n_trials = 10
    test_episodes = 100

    searcher.search(
        param_ranges=param_ranges, n_trials=n_trials, test_episodes=test_episodes
    )

    print("✅ Random search complete!")


if __name__ == "__main__":
    main()
