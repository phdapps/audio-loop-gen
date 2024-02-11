import logging
import numpy as np

from ..util import AudioData, fade_in, fade_out, align_phase

BLEND_DURATION_MS = 50
FADE_IN_DURATION_MS = 500
FADE_IN_MIN_LEVEL = 0.66
FADE_IN_MAX_LEVEL = 1.0
FADE_OUT_DURATION_MS = 800
FADE_OUT_MIN_LEVEL = 0.5
FADE_OUT_MAX_LEVEL = 1.0

class LoopStrategy(object):
    strategy_id: str = None
    
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
    
    def slice_and_blend(self, loop: AudioData, loop_start: int, loop_end: int) -> AudioData:
        """ Slice a loop from the audio data and apply some blending to avoid clicks.

        Args:
            audio (AudioData): The audio data to slice from.
            loop_start (int): The start index of the cut.
            loop_end (int): The end index of the cut.

        Returns:
            AudioData: The sliced and blended audio data.
        """
        # Slice the loop and apply blending
        audio_data = loop.audio_data
        # Quick blend to avoid clicks
        blend_samples = BLEND_DURATION_MS * loop.sample_rate // 1000
        lead_start = max(loop_start - blend_samples // 2, 0)
        blend_end = min(loop_end + blend_samples // 2, audio_data.shape[1])
        lead = audio_data[:, lead_start:lead_start + blend_samples]
        audio_data = audio_data.copy()
        c_in = np.linspace(0, 1, blend_samples)
        c_out = np.linspace(1, 0, blend_samples)
        audio_data[:, blend_end - blend_samples: blend_end] *= c_out
        audio_data[:, blend_end - blend_samples: blend_end] += c_in * lead[:, :]
        
        audio_data = audio_data[:, loop_start:loop_end]
        audio_data = fade_in(audio_data, loop.sample_rate, fade_duration_ms=FADE_IN_DURATION_MS, min_level=FADE_IN_MIN_LEVEL, max_level=FADE_IN_MAX_LEVEL, in_place=True)
        audio_data = fade_out(audio_data, loop.sample_rate, fade_duration_ms=FADE_OUT_DURATION_MS, min_level=FADE_OUT_MIN_LEVEL, max_level=FADE_OUT_MAX_LEVEL, in_place=True)
        
        return AudioData(audio_data, loop.sample_rate)
                