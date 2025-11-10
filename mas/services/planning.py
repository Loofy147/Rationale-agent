import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from mas.data_models import DiscoveryBrief
from mas.prompts import PLAN_GENERATOR_PROMPT

class PlanningService:
    """
    A service for generating an adaptive task plan using a large language model.
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
        print(f"Loading planning model: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Planning model loaded.")

    def generate_plan(self, brief: DiscoveryBrief) -> str:
        """
        Generates a raw markdown task plan from a discovery brief.
        """
        prompt = PLAN_GENERATOR_PROMPT.format(
            topic=brief.topic,
            literature_review=brief.literature_review
        )

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        outputs = pipe(
            prompt,
            max_new_tokens=2048, # Plans can be long
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        return outputs[0]['generated_text']

# planning_service = PlanningService()
