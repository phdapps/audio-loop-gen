import logging

from ..util import AudioData, fade_in, fade_out, equal_power_crossfade, adjust_loop_ends, align_phase

CROSSFADE_DURATION_MS = 600
CROSSFADE_MIN_LEVEL = 0
CROSSFADE_MAX_LEVEL = 0.50
BLEND_FADE_DURATION_MS = CROSSFADE_DURATION_MS * 2
BLEND_FADE_IN_LEVEL = 0.5
BLEND_FADE_OUT_LEVEL = 0.4
PHASE_ALIGN_DURATION_MS = CROSSFADE_DURATION_MS * 2

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
        # Quick blend to avoid clicks
        audio_data = loop.audio_data
        loop_start, loop_end = adjust_loop_ends(loop, loop_start, loop_end)
        
        align_length = min((loop_end - loop_start) // 2, (loop.sample_rate * PHASE_ALIGN_DURATION_MS) // 1000) # PHASE_ALIGN_DURATION_MS or half the loop length
        audio_data = align_phase(audio_data, loop_start, loop_end, segment_length=align_length, in_place=True)
        
        audio_data = audio_data[:, loop_start:loop_end]
        # first crossfade with the start of the loop
        audio_data = equal_power_crossfade(audio_data, loop.sample_rate, crossfade_duration_ms=CROSSFADE_DURATION_MS, max_level=CROSSFADE_MAX_LEVEL, min_level=CROSSFADE_MIN_LEVEL)
        # then apply some fading at start and end
        audio_data = fade_in(audio_data, loop.sample_rate, BLEND_FADE_DURATION_MS, min_level=BLEND_FADE_IN_LEVEL, in_place=True)
        audio_data = fade_out(audio_data, loop.sample_rate, BLEND_FADE_DURATION_MS, min_level=BLEND_FADE_OUT_LEVEL, in_place=True)
        
        return AudioData(audio_data, loop.sample_rate)
                