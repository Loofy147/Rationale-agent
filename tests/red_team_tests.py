import os
import torch
from unittest.mock import MagicMock, patch
from mas.hr_rl.core import get_shaped_reward
from mas.hr_rl.comprehensive_fix import EnhancedDQNAgent

def test_reward_engineering_fix():
    """
    Verifies that a successful outcome is always better than a failure,
    regardless of the number of steps.
    """
    # Arrange: A successful state after many steps
    success_info = {'status': 'SUCCESS'}
    reward_slow_success = get_shaped_reward(0, 100, 100, 90, True, success_info)

    # Arrange: A failure state after few steps
    failure_info = {'status': 'FORBIDDEN'}
    reward_fast_failure = get_shaped_reward(0, 5, 100, 5, True, failure_info)

    # Assert
    assert reward_slow_success > reward_fast_failure, \
        "The reward for a slow success must be greater than for a fast failure."

def test_stage_aware_epsilon_decay():
    """
    Verifies that the agent's epsilon value is reset at new curriculum stages.
    """
    # Arrange
    agent = EnhancedDQNAgent(state_dim=12, action_dim=3)
    initial_epsilon = agent.epsilon

    # Act: Simulate some decay
    for _ in range(10):
        agent.update_epsilon()

    decayed_epsilon = agent.epsilon
    assert decayed_epsilon < initial_epsilon, "Epsilon should decay after updates."

    # Act: Reset epsilon for a new stage
    new_start_value = 0.7
    agent.reset_epsilon(new_start_value)

    # Assert
    assert agent.epsilon == new_start_value, "Epsilon should be reset to the new stage's start value."

@patch('mas.hr_rl.production_run.EnhancedDQNAgent')
def test_checkpointing_system(mock_agent):
    """
    Verifies that the production training script saves model checkpoints.
    """
    # Arrange
    from mas.hr_rl.production_run import train_agent

    # Mock the agent and its network to avoid actual training
    mock_policy_net = MagicMock()
    mock_agent_instance = mock_agent.return_value
    mock_agent_instance.policy_net = mock_policy_net

    # Use a temporary directory for checkpoints
    CHECKPOINT_DIR = "test_checkpoints"

    # Override the configuration in the run script
    with patch('mas.hr_rl.production_run.TRAINING_EPISODES', 2), \
         patch('mas.hr_rl.production_run.CHECKPOINT_FREQ', 1), \
         patch('mas.hr_rl.production_run.CHECKPOINT_DIR', CHECKPOINT_DIR):

        # Act
        train_agent()

    # Assert
    expected_checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_episode_1.pth")
    assert os.path.exists(expected_checkpoint_path), "A periodic checkpoint should have been saved."

    expected_best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(expected_best_model_path), "The best model should have been saved."

    # Clean up
    if os.path.exists(expected_checkpoint_path):
        os.remove(expected_checkpoint_path)
    if os.path.exists(expected_best_model_path):
        os.remove(expected_best_model_path)
    if os.path.exists(CHECKPOINT_DIR):
        os.rmdir(CHECKPOINT_DIR)
