import random
import re
import logging
from typing import Callable, Concatenate
from ..util import LoopGenParams

NUMBERED_LINE_REGEX = re.compile(r"^\d+[\.]?\s+(.*)")

LLM_CHAT_SYSTEM_MESSAGE = \
"""Act as a music expert who can create a wide variety of prompts for music generation.
Your answers should contain 2 parts separated by a pipe character "|", prompt and bpm:
1. `prompt` is the description of the music to generate.  
These are the main parts of a `prompt`:
{parts}

DO always include a melody and instrumentation parts!
DON'T include them all in a single prompt, but choose a few to create a unique prompt!
DON'T generate prompts longer than 160 characters!

2. `bpm` is the beats per minute as an integer between 30 and 150, with 80 percent in the 50-120 range.

You should always follow the set of parameters for generating a melody as described above. They should always be on a single line, i.e. no line breaks, followed by the pip symbol `|` and the `bpm` details!

Example line (make sure all keys like 'prompt:' and 'bpm:' are present! Don't include quotes):
"{example}"
"""

MUSIC_PROMPT_PARTS = [
    """- melody: A description of a melody in 5-10 words. Mandatory! This can be a simple melody or a more complex one. It should reflect the mood and vary significantly from prompt to prompt.""",
    """- instrumentation: 3-10 words. Mandatory! The instruments used in a song. For example, "guitar and drums" or "piano and strings".""",
    """- chord progression: 3-10 words. Optional. The chord progression used in a song. For example, "simple 4-chord progression" or "complex jazz chords".""",
    """- mood and genre: 3-10 words. Optional. Choose from the a wide spectrum of emotions. For examples, "upbeat pop song" or "sad piano ballad", etc.""",
    """- tempo and rhythm: 1-5 words. For example, "slow and steady" or "fast and upbeat". This should also match the `bpm` parameter!""",
    """- musical elements: 3-10 words. For example, "catchy hook" or "haunting melody".""",
    """- musical inspiration: 1-5 words. For example, "inspired by classical music" or "influenced by jazz"."""
]

MUSIC_PROMPT_EXAMPLES = [
    "prompt:upbeat pop song, carefree and happy mood, basic 4-chord progression, guitar and drums with a hint of synth, simple and catchy melody, danceable rhythm|bpm:120"
    "prompt:haunting and melancholic balad, slow chord progression, sad and somber mood, piano and strings, influenced by classical music, slow and steady|bpm:60",
    "prompt:complex and intricate melody, jazzy chord progression, mysterious jazz tune, dark and moody, saxophone and piano with a hint of trumpet. medium, laid-back tempo|bpm:90",
    "prompt:slow and dreamy ambient electronic melody, lush and ethereal chord progression, spacey and atmospheric mood, synth and pads, influenced by EDM|bpm:50"
    "prompt:fast and energetic upbeat rock melody, anthemic and triumphant mood, guitar and drums with a hint of bass. Fast and powerful with a driving rhythm. Inspired by classic rock.|bpm:130",
    "prompt:hip-hop dance beat, simple and repetitive chord progression, upbeat and energetic mood, synth, drums and bass, catchy hook|bpm:100"
]

def trim_line(text:str) -> str:
    # Replace the entire text with the first captured group
    text = strip_space_and_quotes(text)
    text = NUMBERED_LINE_REGEX.sub(r"\1", text)
    return strip_space_and_quotes(text)

def strip_space_and_quotes(text:str) -> str:
    return text.strip(" '`\"")

def randomized_llm_chat_system_message(seed = -1):
    if seed >= 0:
        random.seed(seed)
    # randomize the instructions slightly to avoid getting too similar prompts
    parts = MUSIC_PROMPT_PARTS[:]
    random.shuffle(parts)
    example = random.choice(MUSIC_PROMPT_EXAMPLES)
    return LLM_CHAT_SYSTEM_MESSAGE.format(parts="\n".join(parts), example=example)

class PromptGenerator(object):
    """ Base class for prompt generators.
    """
    def __init__(self, use_case:str = None, params_callback: Callable[Concatenate[str, int, ...], LoopGenParams] = None):
        self.use_case = use_case
        self.params_callback = params_callback if params_callback else lambda prompt, bpm, **kwargs: LoopGenParams(
            prompt=prompt, bpm=bpm, **kwargs)
        self.logger = logging.getLogger("global")
        
    async def generate(self, max_count: int = 1, seed:int = -1) -> list[LoopGenParams]:
        """ Generate generation params for the given number of loops.

        Args:
            max_count (int, optional): The number of loops we want to generate. Defaults to 1.
            kwargs: Additional arguments to pass to the generator.

        Returns:
            list[LoopGenParams]: List of LoopGenParams with size less than or equal to `max_count`.
        """
        raise NotImplementedError