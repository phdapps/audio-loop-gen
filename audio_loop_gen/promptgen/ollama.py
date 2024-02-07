import ollama
from typing import Callable, Concatenate

from .base import PromptGenerator, trim_line, randomized_llm_chat_system_message
from ..util import LoopGenParams

LLAMA_CHAT_USER_MESSAGE_TEMPLATE = "Generate {count} sets of parameters for generating a melody."
LLAMA_CHAT_USER_MESSAGE_TEMPLATE_USE_CASE_EXTRA = "The melody's use case will be \"{use_case}\" so adjust the prompt apropriately but still keep it varied and unique."

class Ollama(PromptGenerator):
    def __init__(self, model_id: str = None, use_case: str = None, params_callback: Callable[Concatenate[str, int, ...], LoopGenParams] = None):
        super().__init__(use_case=use_case, params_callback=params_callback)
        if not model_id:
            model_id = "mistral"
        self.__model_id = model_id
        self.__ollama_client = ollama.AsyncClient()

    async def generate(self, max_count=1, seed: int = -1) -> list[LoopGenParams]:
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

        if seed is None or seed < -1:
            seed = -1

        sys_message = randomized_llm_chat_system_message(seed)

        message = LLAMA_CHAT_USER_MESSAGE_TEMPLATE.format(count=max_count)
        if self.use_case:
            message += " " + \
                LLAMA_CHAT_USER_MESSAGE_TEMPLATE_USE_CASE_EXTRA.format(
                    use_case=self.use_case)
        messages = [{"role": "system", "content": sys_message},
                    {"role": "user", "content": message}]

        response = await self.__ollama_client.chat(
            model=self.__model_id,
            messages=messages,
            options=ollama.Options(seed = seed)
        )

        params_list = []
        message: dict = response.get("message")
        if message:
            content: str = message.get("content")
            if content:
                # We asked the LLM for 1 line per params set
                lines = content.splitlines()
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
                            self.logger.debug(
                                "Invalid llama message: %s", line)
                            continue
                        params = self.params_callback(
                            data.pop("prompt"), int(data.pop("bpm")), **data)
                        params_list.append(params)
                    except Exception as e:
                        # The LLM can generate garbage, which we'll just ignore
                        self.logger.debug(
                            "Error parsing llama message: %s", str(e))
                        continue
        return params_list
