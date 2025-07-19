from transformers import pipeline
from core.diamond_span import DiamondSpan
from core.scarcity_engine import ScarcityEngine

class DiamondLLM:
    def __init__(self, model_path: str):
        self.model = pipeline("text-generation", model=model_path)
        self.scarcity = ScarcityEngine()
    def generate(self, prompt_span: DiamondSpan):
        if not self.scarcity.mint_span(prompt_span):
            raise ValueError("Scarcity limit reached")
        output = self.model(prompt_span.content['text'], max_new_tokens=256)[0]['generated_text']
        response_span = DiamondSpan(
            verb="RESPONDED",
            actor="llm",
            object=prompt_span.id,
            content={"text": output},
            metadata={"creator": "llm", "energy": 10}
        )
        return response_span