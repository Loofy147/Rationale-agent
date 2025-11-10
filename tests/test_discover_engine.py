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
