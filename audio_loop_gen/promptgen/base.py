import random
import re
import logging
from typing import Callable, Concatenate
from ..util import LoopGenParams

NUMBERED_LINE_REGEX = re.compile(r"^\d+[\.]?\s+(.*)")

LLM_CHAT_SYSTEM_MESSAGE = \
"""Act as a music expert who can come up with a wide variety of prompts for music generation using an AI model.

For each music piece you shoult answer with a single line of 2 parts separated by a pipe character "|", the prompt and the bpm:
1. A "prompt" is the description of the music to generate.  
These are the main parts of a prompt:
- melody: A mandatory part that describes the melody in 3-6 words. Mandatory! This can be a simple melody or a more complex one. It should reflect the mood and vary significantly from prompt to prompt.
- instrumentation: A mandatory part listing the 2-3 main instruments playing the music. For example, "guitar and drums" or "piano, strings, and sax".
{parts}

2. "bpm" is the beats per minute as an integer between 30 and 150.

DO always include the `melody` and `instrumentation` parts!
DO always include the prompt and the bpm value on the same line and separated with the pipe symbol "|"!
DO pick 1-2 random optional parts from the list above to create a unique prompt!
DO select the bpm values to match the melody description!
DO prefer bpm values between 50 and 120, 80 percent of bpm values should be in this range!
DO NOT generate prompts longer than 160 characters!
DO NOT include any other information in your answer!
DO NOT use any quotes or backticks in your answer!
DO NOT number the lines in your answer!

Some examples of correct responses:
{examples}
"""

MUSIC_PROMPT_PARTS = [
    """- chord progression: An optional part describing the chord progression in 2-5 words. For example, "simple 4-chord progression" or "complex jazz chords".""",
    """- mood and genre: An optional part listing the 1-3 characterstics of the melody. Choose from the a wide spectrum of human emotions. For examples, "upbeat pop song" or "sad piano ballad", etc.""",
    """- tempo and rhythm: An optional part of 1-3 words. For example, "slow and steady" or "fast and upbeat". This should be reflected in the `bpm` value, i.e. a slow song should have a low bpm!""",
    """- musical elements: An optional part of 1-4 melody qualifiers. For example, "catchy hook" or "haunting melody".""",
    """- musical inspiration: An optional part of 1-4 words. For example, "inspired by classical music" or "influenced by jazz"."""
]

MUSIC_PROMPT_EXAMPLES = [
    "environmentally conscious melody, earthy tones, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves|90",
    "80s electronic track with melodic synthesizers, catchy beat and groovy bass|100",
    "smooth jazz, with a saxophone solo, piano chords, and snare full drums|65",
    "a grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle|120",
    "Rock with saturated guitars, a heavy bass line and crazy drum break and fills|130",
    "Funky and confident, featuring groovy electric guitar, keyboards that create a chill, laid-back mood|80"
    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions|120",
    "acoustic folk song to play during roadtrips with guitar, flute, choirs|80",
    "A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city|100",
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
    examples = random.sample(MUSIC_PROMPT_EXAMPLES, 3)
    return LLM_CHAT_SYSTEM_MESSAGE.format(parts="\n".join(parts), examples="\n".join(examples))

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