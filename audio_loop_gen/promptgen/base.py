from ..util import LoopGenParams

class PromptGenerator(object):
    """ Base class for prompt generators.
    """
    def generate(self, max_count: int = 1) -> list[LoopGenParams]:
        """ Generate generation params for the given number of loops.

        Args:
            count (int, optional): The number of loops we want to generate. Defaults to 1.
            kwargs: Additional arguments to pass to the generator.

        Returns:
            list[LoopGenParams]: List of LoopGenParams with size less than or equal to `max_count`.
        """
        raise NotImplementedError