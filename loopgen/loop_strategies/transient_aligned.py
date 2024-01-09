import numpy as np
from numpy import ndarray
import librosa
import pydub

from .base import LoopStrategy
from ..util import ndarray_to_audio_segment

class TransientAligned(LoopStrategy):
    """
    TransientAligned is a subclass of LoopStrategy specifically designed to create audio loops based on transient 
    detection. Transients are significant, short-duration spikes in the audio signal, often corresponding to 
    percussive or notable sound events.

    The algorithm works by detecting transients within the audio and evaluating if these can form a loop 
    that meets the minimum duration requirements. The primary steps involve:

    1. Transient Detection: Identifying transient points in the audio using onset strength analysis.
    2. Loop Suitability Check: Ensuring there are enough transients to form a loop and that the loop duration 
       meets the minimum requirement.
    """
    MIN_TRANSIENTS_THRESHOLD = 5 # Minimum threshold of transients for loop suitability.
    MAX_TRANSIENTS_THRESHOLD = 15 # Maximum threshold of transients for loop suitability.
    WEIGHT_RMS_ENERGY = 0.6 # Weight for RMS energy in the combined metric.
    WEIGHT_SPECTRAL_FLATNESS = 0.25 # Weight for spectral flatness in the combined metric.
    WEIGHT_DYNAMIC_RANGE = 0.15 # Weight for dynamic range in the combined metric.
    
    def __init__(self, audio_data: ndarray, sample_rate:int=44100, min_loop_duration:int=20000):
        super().__init__(audio_data=audio_data, sample_rate=sample_rate, min_loop_duration=min_loop_duration)
        self.__loop_start:int = -1
        self.__loop_end:int = -1
        self.__evaluated = False
        
    def create_loop(self):
        """
        Create a loop from the audio data based on transient alignment.

        This method relies on the successful evaluation of the audio's suitability for transient-aligned looping.

        Raises:
            ValueError: If the audio is not suitable for transient-aligned looping.

        Returns:
            ndarray: The looped audio data as a NumPy array.
        """
        if not self.evaluate():
            raise ValueError("Audio is not suitable for transient-aligned looping")
        
        data = None
        if self.__is_stereo:  # Stereo audio
            data = self.__audio_data[:, self.__loop_start:self.__loop_end]
        else:  # Mono audio
            data = self.__audio_data[self.__loop_start:self.__loop_end]
        return ndarray_to_audio_segment(data, self.__sample_rate)
        
    def evaluate(self) -> bool:
        if self.__evaluated:
            return self.__loop_start >= 0
        
        audio_data = self.__audio_data
        if self.__is_stereo:
            # Convert stereo audio to mono by averaging the left and right channels
            audio_data = np.mean(audio_data, axis=0)
            
        onset_env = librosa.onset.onset_strength(audio_data, sr = self.__sample_rate)
        transients = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.__sample_rate)
        if transients is None or (not transients.size) or len(transients) < 2:
            self.__evaluated = True
            return False
        threshold = self.__calculate_transients_threshold(audio_data=audio_data)
        suitable = len(transients) > threshold
        if suitable:
            loop_start = librosa.frames_to_samples(transients[0])
            loop_end = librosa.frames_to_samples(transients[-1])
            loop_duration = ((loop_end - loop_start)/self.__sample_rate)*1000 # in milliseconds
            suitable = loop_duration >= self.__min_loop_duration
            
            if suitable:
                self.__loop_start = loop_start
                self.__loop_end = loop_end
        self.__evaluated = True
        
        return suitable
    
    def __calculate_transients_threshold(self, audio_data:ndarray) -> int:
        """Dynamic calculation of the threshold for the number of transients required for the audio data's suitability.

        Returns:
            int: The threshold for the number of transients required for the audio data's suitability.
        """
        # Feature extraction
        rms_energy = np.sqrt(np.mean(np.square(audio_data))) # 0.0 - 1.0
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0].mean() # 0.0 - 1.0
        dynamic_range_norm = (np.max(audio_data) - np.min(audio_data)) / 2 # originally it's between 0.0 - 2.0

        # Weighted combination
        combined_metric = rms_energy * type(self).WEIGHT_RMS_ENERGY + spectral_flatness * type(self).WEIGHT_SPECTRAL_FLATNESS + dynamic_range_norm * type(self).WEIGHT_DYNAMIC_RANGE

        # Map to threshold range (example, adjust based on testing)
        min_threshold = type(self).MIN_TRANSIENTS_THRESHOLD
        max_threshold = type(self).MAX_TRANSIENTS_THRESHOLD
        threshold = int(min_threshold + (max_threshold - min_threshold) * combined_metric)

        return threshold