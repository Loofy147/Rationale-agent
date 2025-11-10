from unittest.mock import patch
from mas.engines.plan import PlanGenerator
from mas.data_models import DiscoveryBrief, AdaptivePlan

# Re-using the valid markdown from the parser test
VALID_MARKDOWN_PLAN = """
# Epic: Test Epic

## Feature: Test Feature 1
| ID | Type | Title | Description | Estimate | Priority | Risk | Dependencies |
|---|---|---|---|---|---|---|---|
| T-1 | Task | Task 1 | Desc 1 | 2h | P0 | L | - |
"""

@patch('mas.engines.plan.PlanningService')
def test_plan_generator_run(mock_planning_service):
    """
    Tests the full workflow of the PlanGenerator with a mocked PlanningService.
    """
    # Arrange
    mock_planning_instance = mock_planning_service.return_value
    mock_planning_instance.generate_plan.return_value = VALID_MARKDOWN_PLAN

    engine = PlanGenerator()
    brief = DiscoveryBrief(topic="test", literature_review="test")

    # Act
    plan = engine.run(brief)

    # Assert
    assert isinstance(plan, AdaptivePlan)
    assert plan.epic_title == "Test Epic"
    assert len(plan.features) == 1
    assert plan.features[0].tasks[0].id == "T-1"

    mock_planning_instance.generate_plan.assert_called_once_with(brief)
