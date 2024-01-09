import numpy as np
from numpy import ndarray
import librosa
import pydub

from .base import LoopStrategy
from ..util import ndarray_to_audio_segment, crossfade

class CrossFade(LoopStrategy):
    """
    A LoopStrategy for generating a loop by crossfading the start and end of the audio data. 
    It evaluates the audio data based on spectral centroids to determine if it's suitable for crossfade looping. 
    This is particularly useful for ambient or textural audio where a seamless loop is desired.
    """
    
    SPECTRAL_CENTROIDS_THRESHOLD = 1000 # Threshold for spectral centroids variance to determine if the audio is ambient or textural
    def __init__(self, audio_data: ndarray, sample_rate:int=44100, min_loop_duration:int=20000, fade_duration:int=1000):
        """
        Initialize the CrossFaded object.

        Parameters:
        audio_data (ndarray): The audio data to be processed.
        sample_rate (int): The sample rate of the audio data, default is 44100 Hz.
        min_loop_duration (int): The minimum duration for the audio loop in milliseconds, default is 20000 ms.
        fade_duration (int): The duration of the fade effect in milliseconds, default is 1000 ms.
        """
        super().__init__(audio_data=audio_data, sample_rate=sample_rate, min_loop_duration=min_loop_duration)
        self.__fade_duration = fade_duration
        self.__is_suitable = None
        
    def evaluate(self) -> bool:
        """
        Evaluate if the audio data is suitable for crossfade looping.

        Returns:
        bool: True if the audio is suitable for crossfade looping, False otherwise.
        """
        if self.__is_suitable is not None:
            return self.__is_suitable
        
        audio_data = self.__audio_data
        if self.__is_stereo:
            # Convert stereo audio to mono by averaging the left and right channels
            audio_data = np.mean(audio_data, axis=0)
        
        spectral_centroids = librosa.feature.spectral_centroid(audio_data, sr=self.__sample_rate)
        self.__is_suitable = np.var(spectral_centroids) < type(self).SPECTRAL_CENTROIDS_THRESHOLD

        return self.__is_suitable

    def create_loop(self) -> pydub.AudioSegment:
        """
        Create a looped version of the audio with a crossfade effect.

        Returns:
        An audio segment with the crossfade effect applied.

        Raises:
        ValueError: If the audio is not suitable for crossfade looping.
        """
        if not self.evaluate():
            raise ValueError("Audio is not suitable for crossfade looping")
        segment = ndarray_to_audio_segment(self.__audio_data, self.__sample_rate)
        return crossfade(segment, self.__fade_duration)