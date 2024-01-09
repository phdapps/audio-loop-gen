import numpy as np
from numpy import ndarray
from pydub import AudioSegment

from .util import audio_segment_to_ndarray 
from .loop_strategies import LoopStrategy, BeatAligned, TransientAligned, CrossFade, FadeInOut

class LoopPipeline:
    def __init__(self, audio_data: ndarray, sample_rate: int, min_loop_duration: int = 30000):
        self.__audio_data = audio_data
        self.__sample_rate = sample_rate
        self.__min_loop_duration = min_loop_duration
        self.__strategies = self.__prepare_strategies()

    def execute(self) -> AudioSegment:
        loop = None
        for strategy in self.__strategies:
            if strategy.evaluate():
                loop = strategy.create_loop()
                break
        if not self.__quality_check(loop):
            loop = self.__adjust_loop(loop)
        
        return loop
    
    def __quality_check(self, loop: AudioSegment):
        """
        Perform a quality check on the loop.
        This function checks for abrupt changes in amplitude which might indicate clicks or pops.
        """
        """
        # Convert to a numpy array for analysis
        loop_np = audio_segment_to_ndarray(loop)

        # Check for large deltas in the waveform (potential clicks or pops)
        delta = np.abs(np.diff(loop_np))
        if np.any(delta > type(self).CLICKS_THRESHOLD): #0.1
            return False
            
        spectrum = np.abs(fft(loop_np))
        spectral_diff = np.abs(np.diff(spectrum))
        if np.any(spectral_diff > SPECTRAL_DIFF_THRESHOLD):
            return False

        # Additional checks can be implemented here
        """
        return True

    def __adjust_loop(self, loop: AudioSegment) -> AudioSegment:
        # Adjust the loop in case of quality check failure
        # This could involve altering loop points, adjusting crossfade, etc.
        return loop
        
    def __prepare_strategies(self) -> list[LoopStrategy]:
        """
        Prepare the ordered list of strategies to be used for looping.
        """
        strategies = []
        # Add the strategies here in the order of preference
        strategies.append(BeatAligned(audio_data=self.__audio_data, sample_rate=self.__sample_rate, min_loop_duration=self.__min_loop_duration))
        strategies.append(TransientAligned(audio_data=self.__audio_data, sample_rate=self.__sample_rate, min_loop_duration=self.__min_loop_duration))
        strategies.append(CrossFade(audio_data=self.__audio_data, sample_rate=self.__sample_rate, min_loop_duration=self.__min_loop_duration))
        strategies.append(FadeInOut(audio_data=self.__audio_data, sample_rate=self.__sample_rate, min_loop_duration=self.__min_loop_duration, fade_duration=1000))
        return strategies