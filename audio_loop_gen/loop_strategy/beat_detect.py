import numpy as np
import librosa
import warnings
from BeatNet.BeatNet import BeatNet

from .base import LoopStrategy
from ..util import AudioData

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) # numpy deprecation warnings from BeatNet

class BeatDetect(LoopStrategy):
    strategy_id: str = "BeatDetect"
    def __init__(self, audio: AudioData, min_loop_duration:int = 20000):
        super().__init__(audio=audio, min_loop_duration=min_loop_duration)
        self.__loop_start: int = -1
        self.__loop_end: int = -1
        self.__evaluated = False
        self.beatnet = BeatNet(
            1,
            mode="offline",
            inference_model="DBN",
            plot=[],
            thread=False,
            device="cuda",
        )

    def evaluate(self) -> bool:
        """Evaluates if the audio is suitable for this loop strategy.

        Returns:
            bool: True if the audio is suitable for this loop strategy, False otherwise.
        """
        if self.__evaluated:
            return self.__loop_start >= 0 and self.__loop_end >= 0
        
        sample_rate = self.audio.sample_rate
        start_time, end_time = -1, -1
        
        # Normalize
        mono_samples = self.audio.mono_audio_data[0]
        normalized_audio = mono_samples / np.abs(mono_samples).max()
        
        try:
            # Estimate beats
            beats = self.__estimate_beats(normalized_audio, sample_rate)
            start_time, end_time = self.__get_loop_points(beats)
        except ValueError:
            self.__evaluated = True
            return False
        
        suitable = start_time >= 0 and end_time >= 0
        if suitable:
            loop_duration = int((end_time - start_time) * 1000) # in milliseconds
            suitable = loop_duration >= self.min_loop_duration
        
        if suitable:
            self.__loop_start = int(start_time * sample_rate)
            self.__loop_end = int(end_time * sample_rate)
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
    
    def __estimate_beats(self, audio_data, sample_rate):
        """ Estimates the beats in the audio data using BeatNet.
        
        Args:
            audio_data (np.ndarray): The audio data.
            sample_rate (int): The sample rate of the audio data.
        Returns:
            np.ndarray: The beat times, shape (N, 2), where N is the number of beats. Each row is a beat time and a beat label (1 for downbeat, 2,3,4 etc. for the rest).
        """
        # BeatNet uses 22050 sample rate internally
        input = librosa.resample(
            audio_data,
            orig_sr=sample_rate,
            target_sr=self.beatnet.sample_rate,
        )
        return self.beatnet.process(input)
    
    def __get_loop_points(self, beats):
        """
        Extracts loop points from beat times.
        
        Args:
            beat_times (np.ndarray): The beat times, shape (N, 2), where N is the number of beats. Each row is a beat time and a beat label (1 for downbeat, 2,3,4 etc. for the rest).
        
        Returns:
            Tuple[float, float]: The start and end time of the loop in seconds.
        """
        # find all downbeats (i.e. the first beat of each bar)
        downbeat_times = beats[:, 0][beats[:, 1] == 1]
        # ... and count the number of complete bars (at least the ones we are sure have completed because we have a downbeat time for the next one)
        # this means we need to ignore the downbeat time for the last bar (as it may not be complete)
        num_bars = len(downbeat_times) - 1

        if num_bars < 1: # one bar would probably be too short, but that will be handled by the min duration check
            raise ValueError("Not enough bars")

        # extract an even number of bars
        even_num_bars = int(2 ** np.floor(np.log2(num_bars)))
        start_time = downbeat_times[0]
        end_time = downbeat_times[even_num_bars]
        
        return start_time, end_time