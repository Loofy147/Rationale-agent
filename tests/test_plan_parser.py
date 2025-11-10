from mas.services.plan_parser import plan_parser_service
from mas.data_models import AdaptivePlan

VALID_MARKDOWN_PLAN = """
# Epic: Test Epic

## Feature: Test Feature 1
| ID | Type | Title | Description | Estimate | Priority | Risk | Dependencies |
|---|---|---|---|---|---|---|---|
| T-1 | Task | Task 1 | Desc 1 | 2h | P0 | L | - |

## Feature: Test Feature 2
| ID | Type | Title | Description | Estimate | Priority | Risk | Dependencies |
|---|---|---|---|---|---|---|---|
| T-2 | Task | Task 2 | Desc 2 | 3d | P1 | M | T-1 |
| T-3 | Spike | Task 3 | Desc 3 | 1d | P0 | H | - |
"""

INVALID_MARKDOWN_PLAN = """
This is not a valid plan.
"""

def test_plan_parser_success():
    """
    Tests that the PlanParserService can successfully parse a valid markdown plan.
    """
    plan = plan_parser_service.parse(VALID_MARKDOWN_PLAN)

    assert isinstance(plan, AdaptivePlan)
    assert plan.epic_title == "Test Epic"
    assert len(plan.features) == 2
    assert plan.features[0].title == "Test Feature 1"
    assert len(plan.features[0].tasks) == 1
    assert plan.features[0].tasks[0].id == "T-1"
    assert plan.features[1].tasks[1].title == "Task 3"

def test_plan_parser_failure():
    """
    Tests that the PlanParserService returns None when parsing invalid markdown.
    """
    plan = plan_parser_service.parse(INVALID_MARKDOWN_PLAN)
    assert plan is None
