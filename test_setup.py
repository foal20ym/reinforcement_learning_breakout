"""
Simple test to verify DQN implementation works correctly.
Run this before starting full training to catch any errors.
"""

import sys
import numpy as np


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        import gymnasium as gym
        import ale_py
        import cv2
        import matplotlib

        print("✓ All packages imported successfully")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_environment():
    """Test that environment can be created and used."""
    print("\nTesting environment...")
    try:
        import gymnasium as gym
        import ale_py

        gym.register_envs(ale_py)

        env = gym.make("ALE/Breakout-v5")
        obs, info = env.reset()
        print(f"✓ Environment created successfully")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Action space: {env.action_space}")

        # Test one step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment error: {e}")
        return False


def test_preprocessing():
    """Test preprocessing wrappers."""
    print("\nTesting preprocessing...")
    try:
        from environment.preprocessing import make_atari_env

        env = make_atari_env("ALE/Breakout-v5")
        obs, info = env.reset()
        print(f"✓ Preprocessing wrappers work")
        print(f"  - Processed observation shape: {obs.shape}")
        print(f"  - Expected shape: (4, 84, 84)")

        assert obs.shape == (4, 84, 84), f"Wrong shape: {obs.shape}"
        print(f"✓ Shape is correct")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False


def test_model():
    """Test DQN model."""
    print("\nTesting DQN model...")
    try:
        import torch
        from core.model import DQNNetwork

        model = DQNNetwork(n_actions=4)
        print(f"✓ Model created successfully")

        # Test forward pass
        dummy_input = torch.zeros(1, 4, 84, 84)
        output = model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected: (1, 4)")

        assert output.shape == (1, 4), f"Wrong output shape: {output.shape}"
        print(f"✓ Output shape is correct")

        return True
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False


def test_replay_buffer():
    """Test replay buffer."""
    print("\nTesting replay buffer...")
    try:
        import numpy as np
        from core.replay_buffer import ReplayBuffer

        buffer = ReplayBuffer(capacity=1000)
        print(f"✓ Replay buffer created")

        # Add some transitions
        for i in range(100):
            state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
            next_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
            buffer.push(state, 0, 1.0, next_state, False)

        print(f"✓ Added 100 transitions")
        print(f"  - Buffer size: {len(buffer)}")

        # Sample batch
        states, actions, rewards, next_states, dones = buffer.sample(32)
        print(f"✓ Sampled batch successfully")
        print(f"  - Batch shapes: states={states.shape}, actions={actions.shape}")

        return True
    except Exception as e:
        print(f"✗ Replay buffer error: {e}")
        return False


def test_agent():
    """Test DQN agent."""
    print("\nTesting DQN agent...")
    try:
        import torch
        from core.agent import DQNAgent
        from utils import config_breakout as config

        agent = DQNAgent(n_actions=4, config=config)
        print(f"✓ Agent created successfully")
        print(f"  - Device: {agent.device}")

        # Test action selection
        state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        action = agent.select_action(state)
        print(f"✓ Action selection works")
        print(f"  - Selected action: {action}")

        # Test storing transition
        next_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        agent.store_transition(state, action, 1.0, next_state, False)
        print(f"✓ Transition storage works")

        return True
    except Exception as e:
        print(f"✗ Agent error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DQN Breakout Implementation Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_environment,
        test_preprocessing,
        test_model,
        test_replay_buffer,
        test_agent,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("✓ All tests passed! Ready to train.")
        print("\nTo start training, run:")
        print("  python main.py --train")
    else:
        print("✗ Some tests failed. Please fix errors before training.")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
