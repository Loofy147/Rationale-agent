import threading
from mas.services.huggingface import hf_search_service
from mas.services.synthesis import SynthesisService
from mas.data_models import DiscoveryBrief

class DiscoverEngine:
    """
    Orchestrates the "Discover" phase of the methodology.
    """
    def __init__(self):
        self.search_service = hf_search_service
        self._synthesis_service = None
        self._synthesis_service_ready = threading.Event()
        self._initialization_thread = threading.Thread(target=self._initialize_synthesis_service)
        self._initialization_thread.start()

    def _initialize_synthesis_service(self):
        """Initializes the synthesis service in a background thread."""
        self._synthesis_service = SynthesisService()
        self._synthesis_service_ready.set()

    @property
    def synthesis_service(self):
        """Waits for the synthesis service to be ready and then returns it."""
        self._synthesis_service_ready.wait()  # Block until the service is ready
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
