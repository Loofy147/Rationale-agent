import os
from unittest.mock import MagicMock
from ..services.planning import PlanningService
from ..services.plan_parser import plan_parser_service
from ..data_models import DiscoveryBrief, AdaptivePlan

# A mock plan to be returned in test mode
MOCK_PLAN_MARKDOWN = """
# Epic: Test E2E Epic

## Feature: Test E2E Feature
| ID | Type | Title | Description | Estimate | Priority | Risk | Dependencies |
|---|---|---|---|---|---|---|---|
| T-1 | Task | E2E Task | Desc | 1d | P0 | L | - |
"""

class PlanGenerator:
    """
    Orchestrates the "Plan" phase of the methodology.
    """
    def __init__(self):
        # In a real application, these services would be injected.
        self.parser_service = plan_parser_service
        # Lazily load the planning service to avoid loading the model on startup.
        self._planning_service = None

    @property
    def planning_service(self):
        if self._planning_service is None:
            if os.getenv("TEST_MODE") == "1":
                # In test mode, use a mock service
                mock_service = MagicMock(spec=PlanningService)
                mock_service.generate_plan.return_value = MOCK_PLAN_MARKDOWN
                self._planning_service = mock_service
            else:
                self._planning_service = PlanningService()
        return self._planning_service

    def run(self, brief: DiscoveryBrief) -> AdaptivePlan | None:
        """
        Runs the full plan generation workflow.
        """
        print(f"Starting plan generation for topic: {brief.topic}")

        # 1. Generate the raw markdown plan from the discovery brief
        raw_plan = self.planning_service.generate_plan(brief)

        # 2. Parse the markdown into a structured object
        structured_plan = self.parser_service.parse(raw_plan)

        if structured_plan:
            print("Plan generation finished successfully.")
        else:
            print("Failed to parse the generated plan.")

        return structured_plan

# Singleton instance of the engine
plan_generator = PlanGenerator()
