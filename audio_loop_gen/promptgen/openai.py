import openai
import os

from .base import PromptGenerator

class OpenAI(PromptGenerator):
    def __init__(self, api_key: str, model_id: str = None):
        # Load the API key from an environment variable
        if api_key is None:
            raise ValueError("Missing API key!")
        
        if not model_id:
            model_id = "gpt-3.5-turbo-1106"
        self.__model_id = model_id
        self.__api_key = api_key
        
        # Configure the OpenAI API key
        openai.api_key = self.__api_key

    def generate_prompts(self, count = 1):
        """
        Query the OpenAI API with the given prompt.
        Args:
            prompt (str): The prompt to send to the API.
            model (str): The model to use for the query. Defaults to "text-davinci-003".
            temperature (float): Controls randomness. Lower is more deterministic.
            max_tokens (int): The maximum number of tokens to generate.
        Returns:
            str: The response from the API.
        """
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].text.strip()
