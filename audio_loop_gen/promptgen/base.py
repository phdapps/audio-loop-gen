import re
from ..util import LoopGenParams

NUMBERED_LINE_REGEX = re.compile(r"^\d+[\.]?\s+(.*)")

def trim_line(text:str) -> str:
    # Replace the entire text with the first captured group
    text = strip_space_and_quotes(text)
    text = NUMBERED_LINE_REGEX.sub(r"\1", text)
    return strip_space_and_quotes(text)

def strip_space_and_quotes(text:str) -> str:
    return text.strip(" '`\"")

class PromptGenerator(object):
    """ Base class for prompt generators.
    """
    
    async def generate(self, max_count: int = 1) -> list[LoopGenParams]:
        """ Generate generation params for the given number of loops.

        Args:
            count (int, optional): The number of loops we want to generate. Defaults to 1.
            kwargs: Additional arguments to pass to the generator.

        Returns:
            list[LoopGenParams]: List of LoopGenParams with size less than or equal to `max_count`.
        """
        raise NotImplementedError