import ollama

from .base import PromptGenerator
from ..util import LoopGenParams

PROMPT_TEMPLATE = """Generate {count} prompts"""

class Ollama(PromptGenerator):
    """_summary_

    Args:
        PromptGenerator (_type_): _description_
    """
    def __init__(self, model_id: str = None):
        # Load the API key from an environment variable
        if not model_id:
            model_id = "gpt-3.5-turbo-1106"
            
    def generate(self, max_count:int = 1) -> list[LoopGenParams]:
        assert max_count is not None and max_count > 0
        llm_prompt = self.__format_llm_prompt(max_count)
        
        return ollama.generate_prompts(max_count)
    
    def __format_llm_prompt(self, count:int = 1) -> str:
        return PROMPT_TEMPLATE.format(count=count)