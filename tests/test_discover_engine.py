import threading
from unittest.mock import patch, MagicMock
from mas.engines.discover import DiscoverEngine
from mas.data_models import DiscoveryBrief

@patch('mas.engines.discover.hf_search_service')
@patch('mas.engines.discover.SynthesisService')
def test_discover_engine_run(mock_synthesis_service, mock_search_service):
    """
    Tests the full workflow of the DiscoverEngine with mocked services.
    """
    # Arrange
    mock_search_service.search_models.return_value = [MagicMock(modelId="test-model")]
    mock_search_service.search_datasets.return_value = [MagicMock(datasetId="test-dataset")]

    mock_synthesis_instance = mock_synthesis_service.return_value
    mock_synthesis_instance.synthesize_discovery_brief.return_value = "This is a synthesized review."

    engine = DiscoverEngine()

    # Act
    brief = engine.run(topic="test topic")

    # Assert
    assert isinstance(brief, DiscoveryBrief)
    assert brief.topic == "test topic"
    assert brief.literature_review == "This is a synthesized review."

    mock_search_service.search_models.assert_called_once_with("test topic")
    mock_search_service.search_datasets.assert_called_once_with("test topic")
    mock_synthesis_instance.synthesize_discovery_brief.assert_called_once()


@patch('mas.engines.discover.hf_search_service')
@patch('mas.engines.discover.SynthesisService')
def test_discover_engine_non_blocking_initialization(mock_synthesis_service, mock_search_service):
    """
    Tests that the DiscoverEngine initializes quickly and the run method waits.
    """
    import time

    # Arrange
    init_event = threading.Event()
    run_event = threading.Event()

    mock_synthesis_instance = MagicMock()
    mock_synthesis_instance.synthesize_discovery_brief.return_value = "A valid review."

    def configured_slow_init(*args, **kwargs):
        init_event.set()
        time.sleep(0.2)
        run_event.set()
        return mock_synthesis_instance

    mock_synthesis_service.side_effect = configured_slow_init

    # Act & Assert: Engine should initialize instantly
    start_time = time.time()
    engine = DiscoverEngine()
    init_time = time.time() - start_time

    assert init_time < 0.1, "Engine initialization should be non-blocking."
    assert init_event.is_set(), "Initialization should have started."
    assert not run_event.is_set(), "Initialization should not have completed yet."

    # Act & Assert: The run method should block until the service is ready
    engine.run(topic="test")

    assert run_event.is_set(), "The run method should have waited for the service to be ready."

    # Clean up the background thread
    engine._initialization_thread.join()
