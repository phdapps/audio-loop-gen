import numpy as np
from numpy import ndarray
import librosa

from .base import LoopStrategy
from ..util import AudioData, find_similar_endpoints, slice_and_blend


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
    MIN_TRANSIENTS_THRESHOLD = 5  # Minimum threshold of transients for loop suitability.
    # Maximum threshold of transients for loop suitability.
    MAX_TRANSIENTS_THRESHOLD = 15
    WEIGHT_RMS_ENERGY = 0.6  # Weight for RMS energy in the combined metric.
    # Weight for spectral flatness in the combined metric.
    WEIGHT_SPECTRAL_FLATNESS = 0.25
    # Weight for dynamic range in the combined metric.
    WEIGHT_DYNAMIC_RANGE = 0.15

    def __init__(self, audio: AudioData, min_loop_duration: int = 20000):
        super().__init__(audio=audio, min_loop_duration=min_loop_duration)
        self.__loop_start: int = -1
        self.__loop_end: int = -1
        self.__evaluated = False

    def evaluate(self) -> bool:
        if self.__evaluated:
            return self.__loop_start >= 0

        audio_data = self.audio.mono_audio_data

        transients = self.__transient_detection(
            audio_data, self.audio.sample_rate)
        if transients is None or transients.size < 2:  # should have at least 2 transients
            self.__evaluated = True
            return False
        threshold = self.__calculate_transients_threshold(
            audio_data=audio_data)
        suitable = len(transients) > threshold
        if suitable:
            loop_start, loop_end = find_similar_endpoints(
                audio_data=audio_data, sample_rate=self.audio.sample_rate, frames=transients)
            if loop_start < 0 or loop_end < 0:
                return False
            loop_duration = ((loop_end - loop_start) /
                             self.audio.sample_rate)*1000  # in milliseconds
            suitable = loop_duration >= self.min_loop_duration

            if suitable:
                self.__loop_start = loop_start
                self.__loop_end = loop_end
        self.__evaluated = True

        return suitable

    def create_loop(self) -> AudioData:
        """
        Create a loop from the audio data based on transient alignment.

        This method relies on the successful evaluation of the audio's suitability for transient-aligned looping.

        Raises:
            ValueError: If the audio is not suitable for transient-aligned looping.

        Returns:
            ndarray: The looped audio data as a NumPy array.
        """
        if not self.evaluate():
            raise ValueError(
                "Audio is not suitable for transient-aligned looping")

        self.logger.debug("Using TransientAligned strategy for loop")
        loop = slice_and_blend(self.audio, self.__loop_start, self.__loop_end)
        return loop

    def __calculate_transients_threshold(self, audio_data: ndarray) -> int:
        """Dynamic calculation of the threshold for the number of transients required for the audio data's suitability.

        Returns:
            int: The threshold for the number of transients required for the audio data's suitability.
        """
        # Feature extraction
        rms_energy = np.sqrt(np.mean(np.square(audio_data)))  # 0.0 - 1.0
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[
            0].mean()  # 0.0 - 1.0
        # originally it's between 0.0 - 2.0
        dynamic_range_norm = (np.max(audio_data) - np.min(audio_data)) / 2

        # Weighted combination
        combined_metric = rms_energy * type(self).WEIGHT_RMS_ENERGY + spectral_flatness * type(
            self).WEIGHT_SPECTRAL_FLATNESS + dynamic_range_norm * type(self).WEIGHT_DYNAMIC_RANGE

        # Map to threshold range (example, adjust based on testing)
        min_threshold = type(self).MIN_TRANSIENTS_THRESHOLD
        max_threshold = type(self).MAX_TRANSIENTS_THRESHOLD
        threshold = int(min_threshold + (max_threshold -
                        min_threshold) * combined_metric)

        return threshold

    def __transient_detection(self, audio_data, sample_rate):
        """Combine onset strength and spectral flux for improved transient detection."""
        # Onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio_data, sr=sample_rate)

        # Spectral flux
        S = np.abs(librosa.stft(audio_data))
        spectral_flux = librosa.onset.onset_strength(
            S=librosa.amplitude_to_db(S, ref=np.max))

        # Combine both metrics
        combined_onset_metric = onset_env + spectral_flux

        # Detect transients
        transients = librosa.onset.onset_detect(
            onset_envelope=combined_onset_metric, sr=sample_rate)

        return transients
