import logging

from ..util import AudioData

class LoopStrategy(object):
    BLEND_SAMPLES = 100
    """A LoopStrategy is used to generate a loop from a given audio segment. 

    It is an implementation of a specific algorithm which can generate a seamless loop from an audio input. 
    A loop strategy which is not suitable for some audio input should return False when its evaluate method is called and raise an exception 
    when its create_loop method is called with such unsuitable audio data.
    """

    def __init__(self, audio: AudioData, min_loop_duration: int = 20000):
        """Basic constructor for a LoopStrategy.

        Args:
            audio (AudioData): The audio data to be used as the base of the generated loop.
            min_loop_duration (int, optional): Minimum duration of the loop that must be generated in ms. Defaults to 20000 (i.e. 20s).
        """
        assert audio is not None
        assert min_loop_duration is not None
        assert min_loop_duration > 0
        
        self.audio: AudioData = audio
        self.min_loop_duration: int = min_loop_duration
        self.logger = logging.getLogger("global")

    def evaluate(self) -> bool:
        """Evaluates if the audio is suitable for the implemented loop strategy.

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Returns:
            bool: True if the audio is suitable for the implemented loop strategy, False otherwise.
        """
        raise NotImplementedError

    def create_loop(self) -> AudioData:
        """Creates a loop using the implemented loop strategy.

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Returns:
            ndarray: The audio data for the loop.
        """
        raise NotImplementedError
                