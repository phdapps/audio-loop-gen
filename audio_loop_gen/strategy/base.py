import numpy as np
from numpy import ndarray
from ..util import AudioData, crossfade, fade_out


class LoopStrategy:
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
        self.audio: AudioData = audio
        self.min_loop_duration: int = min_loop_duration

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
    
    @classmethod
    def slice_and_blend_loop(cls, loop: ndarray, sample_rate: int, loop_start: int, loop_end: int) -> ndarray:
        # Slice the loop and apply blending
        # Quick blend to avoid clicks
        lead_start = max(0, loop_start - cls.BLEND_SAMPLES)
        if loop.ndim == 2 and loop.shape[0] == 2:
            loop = loop[:, loop_start:loop_end]
            lead = loop[:, lead_start:loop_start]
            num_lead = len(lead[0])
            if num_lead > 0:
                loop[:, -num_lead:] *= np.linspace(1, 0, num_lead, dtype=np.float32)
                loop[:, -num_lead:] += np.linspace(0, 1, num_lead, dtype=np.float32) * lead
            else:
                loop = crossfade(loop, sample_rate, 100)
        else:
            loop = loop[loop_start: loop_end]
            lead = loop[lead_start: loop_start]
            num_lead = len(lead)
            if num_lead > 0:
                loop[-num_lead:] *= np.linspace(1, 0, num_lead, dtype=np.float32)
                loop[-num_lead:] += np.linspace(0, 1, num_lead, dtype=np.float32) * lead
            else:
                loop = crossfade(loop, sample_rate, 100)
                
        return loop
                