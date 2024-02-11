import numpy as np
import librosa

from .base import LoopStrategy
from ..util import AudioData

class BeatDetect(LoopStrategy):
    strategy_id: str = "BeatDetect"
    def __init__(self, audio: AudioData, min_loop_duration:int = 20000):
        super().__init__(audio=audio, min_loop_duration=min_loop_duration)
        self.__loop_start: int = -1
        self.__loop_end: int = -1
        self.__evaluated = False

    def evaluate(self) -> bool:
        """Evaluates if the audio is suitable for this loop strategy.

        Returns:
            bool: True if the audio is suitable for this loop strategy, False otherwise.
        """
        if self.__evaluated:
            return self.__loop_start >= 0 and self.__loop_end >= 0
        
        # Normalize
        mono_samples = self.audio.mono_audio_data[0]
        normalized_audio = mono_samples / np.abs(mono_samples).max()

        # Estimate beats
        _, beats = librosa.beat.beat_track(y=normalized_audio, sr=self.audio.sample_rate)
        beat_times = librosa.frames_to_time(beats, sr=self.audio.sample_rate)
        num_bars = len(beat_times) // 4

        # Determine loop points
        start, end = -1, -1
        if num_bars > 0:
            even_num_bars = int(2 ** np.floor(np.log2(num_bars)))
            start =  int(beat_times[0] * self.audio.sample_rate)
            end = int(beat_times[even_num_bars * 4 - 1] * self.audio.sample_rate)
        
        suitable = start >= 0 and end >= 0
        if suitable:
            loop_duration = (end - start) * 1000 / self.audio.sample_rate # in milliseconds
            suitable = loop_duration >= self.min_loop_duration
        
        if suitable:
            self.__loop_start = start
            self.__loop_end = end
        self.__evaluated = True
        
        return suitable

    def create_loop(self) -> AudioData:
        """Creates a loop using the implemented loop strategy.

        Raises:
            ValueError: If the audio is not suitable for this loop strategy.

        Returns:
            AudioData: The audio data for the loop.
        """
        if not self.evaluate():
            raise ValueError("Audio is not suitable for this loop strategy.")
        
        self.logger.debug("Using %s strategy for loop", type(self).strategy_id)
        
        loop = self.slice_and_blend(self.audio, self.__loop_start, self.__loop_end)
        
        return loop