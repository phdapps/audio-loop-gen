import numpy as np
import librosa

from .base import LoopStrategy
from ..util import AudioData, crossfade, fade_out

class CrossFade(LoopStrategy):
    """
    A LoopStrategy for generating a loop by crossfading the start and end of the audio data. 
    It evaluates the audio data based on spectral centroids to determine if it's suitable for crossfade looping. 
    This is particularly useful for ambient or textural audio where a seamless loop is desired.
    """

    # Threshold for spectral centroids variance to determine if the audio is ambient or textural
    SPECTRAL_CENTROIDS_THRESHOLD = 1000

    def __init__(self, audio: AudioData, min_loop_duration: int = 20000):
        """
        Initialize the CrossFaded object.

        Parameters:
        audio_data (ndarray): The audio data to be processed.
        sample_rate (int): The sample rate of the audio data, default is 44100 Hz.
        min_loop_duration (int): The minimum duration for the audio loop in milliseconds, default is 20000 ms.
        fade_duration (int): The duration of the fade effect in milliseconds, default is 1000 ms.
        """
        super().__init__(audio=audio, min_loop_duration=min_loop_duration)
        self.__is_suitable = None

    def evaluate(self) -> bool:
        """
        Evaluate if the audio data is suitable for crossfade looping.

        Returns:
        bool: True if the audio is suitable for crossfade looping, False otherwise.
        """
        if self.__is_suitable is not None:
            return self.__is_suitable

        audio_data = self.audio.mono_audio_data

        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, sr=self.audio.sample_rate)
        self.__is_suitable = np.var(spectral_centroids) < type(
            self).SPECTRAL_CENTROIDS_THRESHOLD

        return self.__is_suitable

    def create_loop(self) -> AudioData:
        """
        Create a looped version of the audio with a crossfade effect.

        Returns:
        An audio segment with the crossfade effect applied.

        Raises:
        ValueError: If the audio is not suitable for crossfade looping.
        """
        if not self.evaluate():
            raise ValueError("Audio is not suitable for crossfade looping")
        print(f"Useing CrossFade strategy for loop")
        loop = crossfade(self.audio.audio_data, self.audio.sample_rate, crossfade_duration_ms=400)
        loop = fade_out(loop, self.audio.sample_rate, fade_duration_ms=600)
        return AudioData(loop, self.audio.sample_rate)