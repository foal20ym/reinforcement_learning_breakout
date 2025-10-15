import gymnasium as gym
import time
import ale_py

gym.register_envs(ale_py)

def play(env, n_episodes=3, max_steps=10000, render=True):
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
            if render:
                # env.render() is implicit when render_mode="human"
                time.sleep(1 / 60)  # slow down for human-visible gameplay
        print(f"Episode {ep} finished in {step} steps, total reward: {total_reward}")

if __name__ == "__main__":
    env = gym.make('ALE/Breakout-v5', render_mode="human")
    try:
        play(env, n_episodes=5)
    finally:
        env.close()