import os

def load_best_avg_reward(filepath = "environment/best_avg_reward.txt"):
    """Load the best average reward from a file."""
    filepath = "environment/best_avg_reward.txt"
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return float(file.read().strip())
    return -float("inf")

def save_best_avg_reward(best_avg_reward):
    """Save the best average reward to a file."""
    filepath = "environment/best_avg_reward.txt"
    with open(filepath, "w") as file:
        file.write(f"{best_avg_reward}")

def load_current_episode(filepath="models/current_episode.txt"):
    """Load last saved episode number (0 if none)."""
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return int(file.read().strip() or 0)
    return 0

def save_current_episode(episode, filepath="models/current_episode.txt"):
    """Save the last completed episode number."""
    with open(filepath, "w") as file:
        file.write(f"{int(episode)}")