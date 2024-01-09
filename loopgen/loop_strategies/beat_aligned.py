import numpy as np
from numpy import ndarray
import librosa
import pydub

from .base import LoopStrategy
from ..util import ndarray_to_audio_segment

class BeatAligned(LoopStrategy):
    """
    BeatAlignedLoop is a loop strategy that aims to create a seamless audio loop by aligning the loop points 
    with the beats of the audio segment. It leverages the principles of beat tracking and tempo analysis in 
    signal processing to find the most rhythmically consistent loop within a given audio segment.

    This strategy works under the assumption that music with a stable tempo and clear beat structure can be 
    looped seamlessly if the loop points are aligned with the beats. The strategy involves two key steps:

    1. Tempo Stability Analysis (__tempo_stability method): 
       This step involves dividing the audio into smaller segments and analyzing the tempo (beats per minute, BPM) 
       of each segment. By calculating the standard deviation of these tempo values across segments, the method 
       assesses the stability of the tempo throughout the audio. A lower standard deviation indicates that the 
       tempo is stable, which is preferable for looping.

    2. Beat Salience Analysis (__beat_salience method):
       This step involves beat tracking to identify the locations of beats within the audio. It then calculates 
       the average strength of these beats to assess their prominence. Strong, clear beats are ideal for creating 
       a loop, as they provide natural points for starting and ending the loop without causing abrupt changes 
       in the rhythm. 

    The loop is considered suitable if the audio has stable tempo and salient beats. The loop points are then 
    determined based on the beat positions, ensuring that the loop starts and ends at these points. Additionally, 
    the loop duration is checked against the minimum loop duration requirement.
    
    IMPORTANT: The source audio duration should be at least 2*SUB_SEGMENT_DURATION, i.e. contain at least 2 sub-segments for analysis.

    Constants:
        SUB_SEGMENT_DURATION (int): Duration (in milliseconds) of each sub-segment for tempo analysis.
        MAX_TEMPO_VARIATION (float): Maximum allowable variation (in BPM) in tempo across segments.
        BEAT_SALIENCE_THRESHOLD (float): Minimum average beat strength for loop suitability.
    """
    SUB_SEGMENT_DURATION = 3000  # Duration of each sub-segment for tempo analysis in milliseconds.
    MAX_TEMPO_VARIATION = 7.5 # Maximum allowed variation in tempo across segments in bpm for loop suitability.
    BEAT_SALIENCE_THRESHOLD = 0.5 # Minimum average beat strength for loop suitability.
    
    def __init__(self, audio_data: ndarray, sample_rate:int=44100, min_loop_duration:int=20000):
        super().__init__(audio_data=audio_data, sample_rate=sample_rate, min_loop_duration=min_loop_duration)
        self.__loop_start:int = -1
        self.__loop_end:int = -1
        self.__evaluated = False

    def evaluate(self) -> bool:
        """
        Evaluate if the audio is suitable for beat-aligned looping.

        This method checks both the beat salience and the tempo stability of the audio.
        The evaluation is performed only once and the result is cached.
        
        Returns:
            bool: True if the audio is suitable for beat-aligned looping, False otherwise.
        """
        if self.__evaluated:
            return self.__loop_start >= 0
        # check the beat salience first to avoid the more computationally expensive tempo stability check
        suitable = self.__beat_salience() and self.__tempo_stability()
        self.__evaluated = True
        return suitable

    def create_loop(self) -> pydub.AudioSegment:
        """
        Create a loop from the audio data based on beat alignment.

        This method relies on the successful evaluation of the audio's suitability for beat-aligned looping.

        Raises:
            ValueError: If the audio is not suitable for beat-aligned looping.

        Returns:
            ndarray: The looped audio data as a NumPy array.
        """
        if not self.evaluate():
            raise ValueError("Audio is not suitable for beat-aligned looping")
        
        data = None
        if self.__is_stereo:  # Stereo audio
            data = self.__audio_data[:, self.__loop_start:self.__loop_end]
        else:  # Mono audio
            data = self.__audio_data[self.__loop_start:self.__loop_end]
        return ndarray_to_audio_segment(data, self.__sample_rate)
    
    def __tempo_stability(self):
        """
        Check the stability of tempo across the audio segment.

        The audio is divided into sub-segments, and the tempo of each segment is analyzed.
        The variation in tempo is then calculated to determine its stability.
        
        Returns:
            bool: True if the variation in tempo is within the acceptable range, False otherwise.
        """
        # Divide the track into smaller segments to detect the tempo for each
        segments_count = int(self.__audio_duration / type(self).SUB_SEGMENT_DURATION)
        if segments_count > 1:
            segment_length = self.__audio_length // segments_count
        else:
            # The track is too short for segmentation and tempo analysis
            return False
            
        tempo_per_segment = []
        # Analyze the tempo for each segment and store in tempo_per_segment.
        for start in range(0, self.__audio_length, segment_length):
            end = min(start + segment_length, self.__audio_length)
            segment = self.__audio_data[start:end]
            if self.__is_stereo:
                # Convert stereo audio to mono by averaging the left and right channels
                segment = np.mean(segment, axis=0)
            segment_tempo, _ = librosa.beat.beat_track(y=segment, sr=self.__sample_rate)
            tempo_per_segment.append(segment_tempo)
            
         # Calculate the standard deviation of the tempo values
        tempo_variation = np.std(tempo_per_segment)
        return tempo_variation <= type(self).MAX_TEMPO_VARIATION  # Threshold for tempo stability

    def __beat_salience(self):
        """
        Evaluate the salience (prominence) of beats in the audio segment.

        Beat tracking is performed on the audio data, and the average strength of the beats is calculated.
        The method also determines if a suitable loop can be formed based on the beat positions and minimum loop duration.
        
        Returns:
            bool: True if beats are salient and a suitable loop duration is achievable, False otherwise.
        """
        audio_data = self.__audio_data
        if self.__is_stereo:
            # Convert stereo audio to mono by averaging the left and right channels
            audio_data = np.mean(audio_data, axis=0)
            
        _, beats = librosa.beat.beat_track(audio_data, sr=self.__sample_rate)
        if not beats or not beats.size:
            return False
        
        # Calculate the average beat strength.
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.__sample_rate)
        avg_beat_strength = np.mean(onset_env[beats])
        
        # Check if the beats are salient and if the loop duration criteria are met.
        if avg_beat_strength > type(self).BEAT_SALIENCE_THRESHOLD:  # Threshold for beat salience
            loop_start = librosa.frames_to_samples(beats[0])
            loop_end = librosa.frames_to_samples(beats[-1])
            min_loop_samples = int((self.__min_loop_duration * self.__sample_rate) / 1000)
            suitable = loop_end - loop_start >= min_loop_samples
            if suitable:
                self.__loop_start = loop_start
                self.__loop_end = loop_end
            return suitable
        else:
            suitable = False
            return False
