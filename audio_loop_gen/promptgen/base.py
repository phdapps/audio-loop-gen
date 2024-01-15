from ..util import LoopGenParams

class PromptGenerator(object):
    """ Base class for prompt generators.
    """
    def generate(self, count: int = 1) -> list[LoopGenParams]:
        """ Generate generation params for the given number of loops.

        Args:
            count (int, optional): The number of loops we want to generate. Defaults to 1.

        Returns:
            list[LoopGenParams]: One params object for each loop we want to generate.
        """
        raise NotImplementedError