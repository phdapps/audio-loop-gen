import openai
import os

from .base import PromptSource

class OpenAI(PromptSource):
    def __init__(self):
        # Load the API key from an environment variable
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        # Configure the OpenAI API key
        openai.api_key = self.api_key

    def query(self, prompt, model="text-davinci-003", temperature=0.7, max_tokens=100):
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

# Example usage
if __name__ == "__main__":
    # Set your API key in the environment variable or directly here for testing purposes
    os.environ['OPENAI_API_KEY'] = 'your-api-key'
    
    import openai
import os

class OpenAIQuery:
    def __init__(self, api_key=None):
        if api_key is None:
            raise ValueError("Missing OpenAI API key!")
        self.api_key = api_key
        
        # Configure the OpenAI API key
        openai.api_key = self.api_key

    def query(self, prompt, model="text-davinci-003", temperature=0.7, max_tokens=100):
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

# Example usage
if __name__ == "__main__":
    # Load the API key from an environment variable
    api_key = os.getenv('OPENAI_API_KEY')

    # Create an instance of the class and make a query
    openai_query = OpenAIQuery(api_key=api_key)
    result = openai_query.query("Translate 'Hello, world!' into French.")
    print(result)