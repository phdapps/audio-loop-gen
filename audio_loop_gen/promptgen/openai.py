import re
from typing import Callable, Concatenate
import openai
import logging

from .base import PromptGenerator, trim_line
from ..util import LoopGenParams

OPENAI_CHAT_COMPLETION_SYSTEM_MESSAGE = {
    "role": "system",
    "content": """You will only answer with a set of parameters for generating a melody as described bellow. It should always be on a single line, i.e. no line breaks!
`prompt`:  The description of the music to generate.  
These are the main parts of a `prompt`:
- melody: A description of a melody in 5-10 words. This can be a simple melody or a more complex one. It should reflect the mood and vary significantly from prompt to prompt.
- chord progression: 3-10 words.
- mood and genre: 3-10 words. Choose from the a wide spectrum of emotions. For examples, "upbeat pop song" or "sad piano ballad", etc. 
- instrumentation: 1-5 words. The instruments used in a song. For example, "guitar and drums" or "piano and strings". 
- tempo and rhythm: 1-5 words. For example, "slow and steady" or "fast and upbeat". This should also match the `bpm` parameter! 
- musical elements: 3-10 words. For example, "catchy hook" or "haunting melody". 
- musical inspiration: 3-10 words. For example, "inspired by classical music" or "influenced by jazz".
`bpm`: The beats per minute as an integer between 30 and 150, with under 120 being 80 percent of generated values.

Example set:
prompt:A simple and catchy melody that is easy to hum along to. A basic 4-chord progression that repeats throughout the song. Upbeat pop song with a carefree and happy mood. Guitar and drums with a hint of synth. Fast and upbeat with a danceable rhythm. Catchy hook.|bpm:120
"""}

OPENAI_CHAT_COMPLETION_USER_MESSAGE_TEMPLATE = "Generate {count} sets of parameters for generating a melody."
OPENAI_CHAT_COMPLETION_USER_MESSAGE_TEMPLATE_USE_CASE_EXTRA =  "The melody's use case will be \"{use_case}\" so adjust the prompt apropriately."

TRIM_LINE_NUM_REGEX = r"^\d+\s+"
class OpenAI(PromptGenerator):
    def __init__(self, api_key: str, model_id: str = None, use_case:str = None, params_callback: Callable[Concatenate[str, int, ...], LoopGenParams] = None):
        super().__init__(use_case=use_case, params_callback = params_callback)
        if api_key is None:
            raise ValueError("Missing API key!")

        if not model_id:
            model_id = "gpt-3.5-turbo-1106"
        self.__model_id = model_id
        self.__openai_client = openai.AsyncOpenAI(api_key=api_key)

    async def generate(self, max_count=1) -> list[LoopGenParams]:
        """
        Query the OpenAI API with the given prompt.
        Args:
            prompt (str): The prompt to send to the API.
            model (str): The model to use for the query. Defaults to "text-davinci-003".
            temperature (float): Controls randomness. Lower is more deterministic.
            max_tokens (int): The maximum number of tokens to generate.
        Returns:
            str: The response from the API. It will be a list of LoopGenParams, with size less than or equal to `max_tokens`.
        """
        assert max_count is not None and max_count > 0
        if max_count < 1:
            max_count = 1
            
        message = OPENAI_CHAT_COMPLETION_USER_MESSAGE_TEMPLATE.format(count=max_count)
        if self.use_case:
            message += " " + OPENAI_CHAT_COMPLETION_USER_MESSAGE_TEMPLATE_USE_CASE_EXTRA.format(use_case=self.use_case)
        messages = [OPENAI_CHAT_COMPLETION_SYSTEM_MESSAGE, {
            "role": "user", "content": message}]

        response = await self.__openai_client.chat.completions.create(
            model=self.__model_id,
            messages=messages
        )

        params_list = []
        if len(response.choices) > 0:
            for choice in response.choices:
                reason = choice.finish_reason
                if reason != "stop" and reason != "length":
                    continue
                response_message = choice.message.content
                if not response_message:
                    continue
                # We asked the LLM for 1 line per params set
                lines = response_message.splitlines()
                for line in lines:
                    line = trim_line(line)
                    # each line should have the format: "prompt:<prompt>|bpm:<bpm>|other_key:other_value..."
                    try:
                        data = {}
                        parts = line.split("|")
                        for part in parts:
                            kvs = part.split(":", maxsplit=1)
                            data[kvs[0]] = kvs[1]
                        if not "prompt" in data or not "bpm" in data:
                            self.logger.debug("Invalid OpenAI completion: %s", line)
                            continue
                        params = self.params_callback(data.pop("prompt"), int(data.pop("bpm")), **data)
                        params_list.append(params)
                    except Exception as e:
                        # The LLM can generate garbage, which we'll just ignore
                        self.logger.debug("Error parsing OpenAI completion: %s", str(e))
                        continue
        return params_list
