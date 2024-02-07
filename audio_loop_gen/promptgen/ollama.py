import random
import ollama
from typing import Callable, Concatenate

import logging

from .base import PromptGenerator, trim_line
from ..util import LoopGenParams


LLAMA_CHAT_SYSTEM_MESSAGE = \
    """Act as a music expert who can create a wide variety of prompts for music generation.
Your answers should contain 2 parts separated by a pipe character "|", prompt and bpm:
1. `prompt` is the description of the music to generate.  
These are the main parts of a `prompt`:
{parts}

2. `bpm` is the beats per minute as an integer between 30 and 150, with 80 percent in the 50-120 range.

You should always follow the set of parameters for generating a melody as described above. They should always be on a single line, i.e. no line breaks, followed by the pip symbol `|` and the `bpm` details!

Example line (make sure all keys like 'prompt:' and 'bpm:' are present! Don't include quotes):
"{example}"
"""

PROMPT_PARTS = [
    """- melody: A description of a melody in 5-10 words. This can be a simple melody or a more complex one. It should reflect the mood and vary significantly from prompt to prompt.""",
    """- chord progression: 3-10 words.""",
    """- mood and genre: 3-10 words. Choose from the a wide spectrum of emotions. For examples, "upbeat pop song" or "sad piano ballad", etc.""",
    """- instrumentation: 3-10 words. The instruments used in a song. For example, "guitar and drums" or "piano and strings".""",
    """- tempo and rhythm: 1-5 words. For example, "slow and steady" or "fast and upbeat". This should also match the `bpm` parameter!""",
    """- musical elements: 3-10 words. For example, "catchy hook" or "haunting melody".""",
    """- musical inspiration: 1-5 words. For example, "inspired by classical music" or "influenced by jazz"."""
]

PROMPT_EXAMPLES = [
    "prompt:A simple and catchy melody that is easy to hum along to. A basic 4-chord progression that repeats throughout the song. Upbeat pop song with a carefree and happy mood. Guitar and drums with a hint of synth. Fast and upbeat with a danceable rhythm. Catchy hook.|bpm:120"
    "prompt:A haunting and melancholic melody that evokes a sense of longing. A slow and sad chord progression that tugs at the heartstrings. Sad piano ballad with a somber and reflective mood. Piano and strings with a touch of cello. Slow and steady with a mournful rhythm. Influenced by classical music.|bpm:60",
    "prompt:A complex and intricate melody that is full of twists and turns. A jazzy chord progression that keeps you guessing. Mysterious jazz tune with a dark and moody atmosphere. Saxophone and piano with a hint of trumpet. Medium tempo with a laid-back rhythm. Inspired by modern jazz.|bpm:90",
    "prompt:A slow and dreamy melody that transports you to another world. A lush and ethereal chord progression that creates a sense of wonder. Ambient electronic song with a spacey and atmospheric mood. Synth and pads with a touch of arpeggios. Slow and floating with an otherworldly rhythm. Haunting melody.|bpm:70"
    "prompt:A fast and energetic melody that gets your heart racing. A driving chord progression that propels the song forward. Upbeat rock song with an anthemic and triumphant mood. Guitar and drums with a hint of bass. Fast and powerful with a driving rhythm. Inspired by classic rock.|bpm:130"
]

LLAMA_CHAT_USER_MESSAGE_TEMPLATE = "Generate {count} sets of parameters for generating a melody."
LLAMA_CHAT_USER_MESSAGE_TEMPLATE_USE_CASE_EXTRA = "The melody's use case will be \"{use_case}\" so adjust the prompt apropriately but still keep it varied and unique."


class Ollama(PromptGenerator):
    def __init__(self, model_id: str = None, use_case: str = None, params_callback: Callable[Concatenate[str, int, ...], LoopGenParams] = None):
        super().__init__(use_case=use_case, params_callback=params_callback)
        if not model_id:
            model_id = "mistral"
        self.__model_id = model_id
        self.__ollama_client = ollama.AsyncClient()

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

        # randomize the instructions slightly to avoid getting too similar prompts
        parts = PROMPT_PARTS[:]
        random.shuffle(parts)
        example = random.choice(PROMPT_EXAMPLES)
        sys_message = LLAMA_CHAT_SYSTEM_MESSAGE.format(parts="\n".join(parts), example=example)

        message = LLAMA_CHAT_USER_MESSAGE_TEMPLATE.format(count=max_count)
        if self.use_case:
            message += " " + \
                LLAMA_CHAT_USER_MESSAGE_TEMPLATE_USE_CASE_EXTRA.format(
                    use_case=self.use_case)
        messages = [{"role": "system", "content": sys_message},
                    {"role": "user", "content": message}]

        response = await self.__ollama_client.chat(
            model=self.__model_id,
            messages=messages
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
