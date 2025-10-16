import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class Logger:
    """
    Logger for tracking training metrics.
    """

    def __init__(self, log_dir="logs"):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilons = []

        # Running average
        self.reward_window = deque(maxlen=100)

    def log_episode(self, episode, reward, length, epsilon, avg_loss=None):
        """
        Log episode statistics.

        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length
            epsilon: Current epsilon value
            avg_loss: Average loss during episode
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilons.append(epsilon)
        self.reward_window.append(reward)

        if avg_loss is not None:
            self.losses.append(avg_loss)

        avg_reward = np.mean(self.reward_window)

        print(
            f"Episode {episode} | Reward: {reward:.2f} | Avg(100): {avg_reward:.2f} | "
            f"Length: {length} | Epsilon: {epsilon:.3f}",
            end="",
        )
        if avg_loss is not None:
            print(f" | Loss: {avg_loss:.4f}")
        else:
            print()

    def plot_training(self, save_path=None):
        """
        Plot training curves.

        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6)
        if len(self.episode_rewards) >= 100:
            avg_rewards = [
                np.mean(self.episode_rewards[max(0, i - 99) : i + 1]) for i in range(len(self.episode_rewards))
            ]
            axes[0, 0].plot(avg_rewards, linewidth=2, label="Average (100 episodes)")
            axes[0, 0].legend()
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].grid(True)

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.6)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].grid(True)

        # Epsilon decay
        axes[1, 0].plot(self.epsilons)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Epsilon")
        axes[1, 0].set_title("Exploration Rate (Epsilon)")
        axes[1, 0].grid(True)

        # Loss
        if self.losses:
            axes[1, 1].plot(self.losses, alpha=0.6)
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].set_title("Training Loss")
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Training plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.log_dir, "training_plot.png"))
            print(f"Training plot saved to {os.path.join(self.log_dir, 'training_plot.png')}")

        plt.close()

    def save_stats(self):
        """Save statistics to file."""
        stats_file = os.path.join(self.log_dir, "training_stats.npz")
        np.savez(
            stats_file,
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            losses=self.losses,
            epsilons=self.epsilons,
        )
        print(f"Statistics saved to {stats_file}")
