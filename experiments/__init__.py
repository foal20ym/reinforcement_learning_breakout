"""
Experiment framework for testing reward shaping configurations.
"""

from .experiment_config import EXPERIMENT_CONFIGS, QUICK_TEST_EPISODES, FULL_TEST_EPISODES
from .experiment_runner import ExperimentRunner

__all__ = ["EXPERIMENT_CONFIGS", "QUICK_TEST_EPISODES", "FULL_TEST_EPISODES", "ExperimentRunner"]
