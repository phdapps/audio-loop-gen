import numpy as np
from numpy import ndarray
import pydub

class LoopStrategy:
    """A LoopStrategy is used to generate a loop from a given audio segment. 
    
    It is an implementation of a specific algorithm which can generate a seamless loop from an audio input. 
    A loop strategy which is not suitable for some audio input should return False when its evaluate method is called and raise an exception 
    when its create_loop method is called with such unsuitable audio data.
    """
    def __init__(self, audio_data:ndarray, sample_rate:int=44100, min_loop_duration:int=20000):
        """Basic constructor for a LoopStrategy.

        Args:
            audio_data (ndarray): The audio data to be used as the base of the generated loop.
            sample_rate (int, optional): The sample rate of the audio data. Defaults to 44100.
            min_loop_duration (int, optional): Minimum duration of the loop that must be generated in ms. Defaults to 20000 (i.e. 20s).
        """
        self.__audio_data:ndarray = audio_data
        self.__is_stereo = audio_data.ndim == 2 and audio_data.shape[0] == 2
        if self.__is_stereo:
            self.__audio_length:int = audio_data.shape[1]
        else:
            self.__audio_length:int = len(audio_data)
            
        self.__sample_rate:int = sample_rate
        self.__min_loop_duration:int = min_loop_duration
        self.__audio_duration:int = (self.__audio_length * 1000) // sample_rate # in milliseconds
        
    def evaluate(self) -> bool:
        """Evaluates if the audio is suitable for the implemented loop strategy.

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Returns:
            bool: True if the audio is suitable for the implemented loop strategy, False otherwise.
        """
        raise NotImplementedError

    def create_loop(self) -> pydub.AudioSegment:
        """Creates a loop using the implemented loop strategy.

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Returns:
            ndarray: The audio data for the loop.
        """
        raise NotImplementedError