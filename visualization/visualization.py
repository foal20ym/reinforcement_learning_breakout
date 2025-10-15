import matplotlib.pyplot as plt

def plot_progress(rewards_per_episode, epsilon_history):
        plt.figure(1)
        plt.subplot(121)
        plt.plot(rewards_per_episode)
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay")
        plt.savefig('plots/breakout_dqn.png')
