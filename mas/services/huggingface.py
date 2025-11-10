from huggingface_hub import HfApi

class HuggingFaceSearchService:
    """
    A service to interact with the Hugging Face Hub.
    """
    def __init__(self):
        self.api = HfApi()

    def search_models(self, query: str, top_n: int = 5):
        """
        Searches for models on the Hugging Face Hub.
        """
        models = self.api.list_models(
            filter=query,
            sort="downloads",
            direction=-1,
            limit=top_n
        )
        return list(models)

    def search_datasets(self, query: str, top_n: int = 5):
        """
        Searches for datasets on the Hugging Face Hub.
        """
        datasets = self.api.list_datasets(
            filter=query,
            sort="downloads",
            direction=-1,
            limit=top_n
        )
        return list(datasets)

# Singleton instance of the service
hf_search_service = HuggingFaceSearchService()
