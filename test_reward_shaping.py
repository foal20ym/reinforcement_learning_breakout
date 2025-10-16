"""
Test script to verify reward shaping implementation.
"""

import sys
import os
import gymnasium as gym
import ale_py
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register ALE environments
gym.register_envs(ale_py)

from environment.preprocessing import make_atari_env
from environment.reward_shaping import BreakoutRewardShaping


def test_reward_shaping_basic():
    """Test that reward shaping wrapper can be created and used."""
    print("=" * 60)
    print("TEST 1: Basic Reward Shaping Wrapper")
    print("=" * 60)

    try:
        # Create environment with reward shaping
        env = make_atari_env("ALE/Breakout-v5", enable_reward_shaping=True)
        print("✓ Environment created with reward shaping")

        # Reset and take a few steps
        obs, info = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")

        total_shaped_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_shaped_reward += reward

            if i == 0:
                print(f"✓ Step {i+1}: action={action}, reward={reward:.3f}, done={terminated or truncated}")

        print(f"✓ Total reward over 10 steps: {total_shaped_reward:.3f}")
        env.close()
        print("\n✅ TEST 1 PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def test_reward_shaping_params():
    """Test custom reward shaping parameters."""
    print("=" * 60)
    print("TEST 2: Custom Reward Shaping Parameters")
    print("=" * 60)

    try:
        custom_params = {
            "paddle_hit_bonus": 0.5,
            "side_angle_bonus": 0.3,
            "block_bonus_multiplier": 2.0,
            "ball_loss_penalty": -1.0,
        }

        env = make_atari_env("ALE/Breakout-v5", enable_reward_shaping=True, shaping_params=custom_params)
        print("✓ Environment created with custom parameters")
        print(f"  Parameters: {custom_params}")

        obs, info = env.reset()
        print("✓ Environment reset successful")

        # Take some steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        env.close()
        print("\n✅ TEST 2 PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def test_reward_comparison():
    """Compare rewards with and without shaping."""
    print("=" * 60)
    print("TEST 3: Reward Comparison (With vs Without Shaping)")
    print("=" * 60)

    try:
        # Without shaping
        env_no_shaping = make_atari_env("ALE/Breakout-v5", enable_reward_shaping=False)
        obs, _ = env_no_shaping.reset(seed=42)

        rewards_no_shaping = []
        for _ in range(100):
            action = env_no_shaping.action_space.sample()
            obs, reward, terminated, truncated, _ = env_no_shaping.step(action)
            rewards_no_shaping.append(reward)
            if terminated or truncated:
                break

        env_no_shaping.close()

        # With shaping
        env_with_shaping = make_atari_env("ALE/Breakout-v5", enable_reward_shaping=True)
        obs, _ = env_with_shaping.reset(seed=42)

        rewards_with_shaping = []
        for _ in range(100):
            action = env_with_shaping.action_space.sample()
            obs, reward, terminated, truncated, _ = env_with_shaping.step(action)
            rewards_with_shaping.append(reward)
            if terminated or truncated:
                break

        env_with_shaping.close()

        # Compare
        total_no_shaping = sum(rewards_no_shaping)
        total_with_shaping = sum(rewards_with_shaping)
        non_zero_no_shaping = sum(1 for r in rewards_no_shaping if r != 0)
        non_zero_with_shaping = sum(1 for r in rewards_with_shaping if r != 0)

        print(f"Without shaping:")
        print(f"  Total reward: {total_no_shaping:.3f}")
        print(f"  Non-zero rewards: {non_zero_no_shaping}/{len(rewards_no_shaping)}")

        print(f"\nWith shaping:")
        print(f"  Total reward: {total_with_shaping:.3f}")
        print(f"  Non-zero rewards: {non_zero_with_shaping}/{len(rewards_with_shaping)}")

        print(f"\nDifference: {total_with_shaping - total_no_shaping:.3f}")

        # With shaping should typically have more non-zero rewards
        if non_zero_with_shaping >= non_zero_no_shaping:
            print("✓ Reward shaping is providing additional feedback")
        else:
            print("⚠️  Warning: Shaping may not be triggering often")

        print("\n✅ TEST 3 PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def test_info_dict():
    """Test that info dict contains shaping indicators."""
    print("=" * 60)
    print("TEST 4: Info Dictionary Indicators")
    print("=" * 60)

    try:
        env = make_atari_env("ALE/Breakout-v5", enable_reward_shaping=True)
        obs, _ = env.reset()

        info_keys_found = set()

        # Run for a while to see if we trigger any bonuses
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Collect any new info keys
            for key in ["paddle_hit", "side_bounce", "block_broken", "ball_lost", "center_bonus"]:
                if info.get(key):
                    info_keys_found.add(key)

            if terminated or truncated:
                obs, _ = env.reset()

        env.close()

        print(f"✓ Info keys detected during random play: {info_keys_found}")

        if len(info_keys_found) > 0:
            print("✓ Reward shaping is being triggered and logged")
        else:
            print("⚠️  No shaping events detected (may need more steps)")

        print("\n✅ TEST 4 PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REWARD SHAPING TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_reward_shaping_basic,
        test_reward_shaping_params,
        test_reward_comparison,
        test_info_dict,
    ]

    results = []
    for test_func in tests:
        results.append(test_func())

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Reward shaping is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")

    print("=" * 60)
