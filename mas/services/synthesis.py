import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class SynthesisService:
    """
    A service for synthesizing text using a large language model.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the tokenizer and model.
        """
        # In a real application, you would add error handling and logging here.
        # This can be slow and memory-intensive, and would be done in a separate
        # worker process in a production system.
        print(f"Loading synthesis model: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Synthesis model loaded.")

    def synthesize_discovery_brief(self, topic: str, models: list, datasets: list) -> str:
        """
        Generates a discovery brief summary from a topic and search results.
        """
        model_list = "\n".join([f"- {m.modelId} (Downloads: {m.downloads})" for m in models])
        dataset_list = "\n".join([f"- {d.datasetId} (Downloads: {d.downloads})" for d in datasets])

        prompt = f"""
        As an expert AI engineer, write a "Literature and Ecosystem Review" for a project discovery brief.

        The project topic is: "{topic}"

        I have found the following top models and datasets from the Hugging Face Hub:

        **Top Models:**
        {model_list}

        **Top Datasets:**
        {dataset_list}

        Based on this information, write a brief, coherent summary for the "Literature and Ecosystem Review" section of the discovery brief. Your summary should highlight the key models and technologies in this area.
        """

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        outputs = pipe(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        return outputs[0]['generated_text']

# Note: This is a heavyweight object. In a real application, we would manage its
# lifecycle carefully and likely run it in a separate process.
# For the MVP, we can instantiate it globally, but this is not ideal for production.
# synthesis_service = SynthesisService()
