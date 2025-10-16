import os
import ale_py
import gymnasium as gym
import numpy as np

# Register ALE environments
gym.register_envs(ale_py)

# Import our modules
from core.agent import DQNAgent
from environment.preprocessing import make_atari_env
from utils import config_breakout as config


def evaluate(checkpoint_path, n_episodes=10, render=True):
    """
    Evaluate a trained DQN agent.

    Args:
        checkpoint_path: Path to the checkpoint file
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    """
    # Note: Reward shaping is NOT used during evaluation
    # We always evaluate with the true environment rewards

    # Create environment
    if render:
        env = gym.make(config.ENV_NAME, render_mode="human")
        # Apply same preprocessing
        from environment.preprocessing import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, ProcessFrame84, FrameStack

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ProcessFrame84(env)
        env = FrameStack(env, k=4)
    else:
        # Do NOT enable reward shaping during evaluation
        env = make_atari_env(config.ENV_NAME, enable_reward_shaping=False)

    n_actions = env.action_space.n

    # Create agent
    agent = DQNAgent(n_actions, config)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        agent.epsilon = 0.05  # Small epsilon for evaluation
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Evaluation loop
    episode_rewards = []
    episode_lengths = []

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        done = False
        while not done:
            # Select action (with minimal exploration)
            action = agent.select_action(state, training=False)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # Print statistics
    print("\n" + "=" * 50)
    print(f"Evaluation over {n_episodes} episodes:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DQN agent on Breakout")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/dqn_breakout_final.pt", help="Path to checkpoint file"
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, render=not args.no_render)
