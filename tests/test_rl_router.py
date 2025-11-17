from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

@patch('mas.routers.rl.rl_runner')
def test_start_training_and_evaluation(mock_rl_runner):
    """
    Tests that the /rl/start-training endpoint returns 202 and
    starts the training in the background.
    """
    # Arrange
    from mas.main import app
    client = TestClient(app)

    # The background task will run in the test's event loop, so we don't
    # need to mock BackgroundTasks. We just need to mock the functions it calls.
    mock_rl_runner.train_agent.return_value = MagicMock() # Mock the returned agent

    # Act
    response = client.post("/rl/start-training")

    # Assert
    assert response.status_code == 202
    assert response.json() == {"message": "Reinforcement learning training and evaluation started in the background."}

    # Verify that the background task called our runner functions
    mock_rl_runner.train_agent.assert_called_once()
    mock_rl_runner.evaluate_agent.assert_called_once()
