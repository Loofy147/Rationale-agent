from unittest.mock import Mock, patch
from mas.services.huggingface import HuggingFaceSearchService
from mas.data_models import DiscoveryBrief

def test_discovery_brief_model():
    """
    Tests the DiscoveryBrief data model.
    """
    brief = DiscoveryBrief(
        topic="test topic",
        literature_review="This is a test review."
    )
    assert brief.type == "discovery_brief"
    assert brief.topic == "test topic"
    assert brief.model_dump()["literature_review"] == "This is a test review."

@patch('mas.services.huggingface.HfApi')
def test_huggingface_search_service(mock_hf_api):
    """
    Tests the HuggingFaceSearchService with a mocked HfApi.
    """
    # Arrange
    mock_api_instance = mock_hf_api.return_value
    # The HfApi returns an iterator, so we mock it to return an iterator.
    mock_api_instance.list_models.return_value = iter([Mock(modelId="test-model")])
    mock_api_instance.list_datasets.return_value = iter([Mock(datasetId="test-dataset")])

    service = HuggingFaceSearchService()

    # Act
    models = service.search_models("test-query")
    datasets = service.search_datasets("test-query")

    # Assert
    assert len(models) == 1
    assert models[0].modelId == "test-model"
    mock_api_instance.list_models.assert_called_once_with(
        filter="test-query",
        sort="downloads",
        direction=-1,
        limit=5
    )

    assert len(datasets) == 1
    assert datasets[0].datasetId == "test-dataset"
    mock_api_instance.list_datasets.assert_called_once_with(
        filter="test-query",
        sort="downloads",
        direction=-1,
        limit=5
    )
