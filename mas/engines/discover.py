import os
from unittest.mock import MagicMock
from ..services.huggingface import hf_search_service
from ..services.synthesis import SynthesisService
from ..data_models import DiscoveryBrief

class DiscoverEngine:
    """
    Orchestrates the "Discover" phase of the methodology.
    """
    def __init__(self):
        # In a real application, these services would be injected.
        self.search_service = hf_search_service
        # Lazily load the synthesis service to avoid loading the model on startup.
        self._synthesis_service = None

    @property
    def synthesis_service(self):
        if self._synthesis_service is None:
            if os.getenv("TEST_MODE") == "1":
                # In test mode, use a mock service
                mock_service = MagicMock(spec=SynthesisService)
                mock_service.synthesize_discovery_brief.return_value = "Mocked literature review."
                self._synthesis_service = mock_service
            else:
                self._synthesis_service = SynthesisService()
        return self._synthesis_service

    def run(self, topic: str) -> DiscoveryBrief:
        """
        Runs the full discovery workflow.
        """
        print(f"Starting discovery for topic: {topic}")

        # 1. Search for models and datasets
        models = self.search_service.search_models(topic)
        datasets = self.search_service.search_datasets(topic)

        # 2. Synthesize the results into a brief
        literature_review = self.synthesis_service.synthesize_discovery_brief(
            topic=topic,
            models=models,
            datasets=datasets
        )

        # 3. Create the structured artifact
        brief = DiscoveryBrief(
            topic=topic,
            literature_review=literature_review
        )

        print("Discovery finished.")
        return brief

# Singleton instance of the engine
discover_engine = DiscoverEngine()
