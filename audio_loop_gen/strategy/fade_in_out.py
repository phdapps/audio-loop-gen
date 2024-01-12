from .base import LoopStrategy
from ..util import AudioData, fade_in, fade_out

class FadeInOut(LoopStrategy):
    """
    SimpleFadeLoop is a subclass of LoopStrategy designed to create audio loops using a simple fade-in/fade-out technique.
    This method is a catch-all strategy suitable for a wide range of audio types, especially when other strategies fail.

    The primary steps involve applying a fade-out effect at the end and a fade-in effect at the beginning of the audio.
    The duration of these fades can be adjusted to suit different audio lengths and types.
    """

    def __init__(self, audio: AudioData, min_loop_duration: int = 20000, fade_duration: int = 200):
        """
        Initialize the SimpleFadeLoop object.

        Parameters:
        audio_data (ndarray): The audio data to be processed.
        sample_rate (int): The sample rate of the audio data, default is 44100 Hz.
        min_loop_duration (int): The minimum duration for the audio loop in milliseconds, default is 20000 ms.
        fade_duration (int): The duration of the fade effect in milliseconds, default is 200 ms.
        """
        super().__init__(audio=audio, min_loop_duration=min_loop_duration)

        self.__fade_duration: int = min(
            fade_duration, self.audio.duration // 3)

    def evaluate(self) -> bool:
        """
        This strategy is suitable for any audio
        """
        return True

    def create_loop(self) -> AudioData:
        """
        Create a looped version of the audio with a simple fade-in/fade-out effect.

        Returns:
        An audio segment with the fade-in/fade-out effect applied.

        Raises:
        ValueError: If the audio is not suitable for simple fade looping.
        """
        print("Using FadeInOut strategy for loop")
        loop = fade_in(self.audio.audio_data, self.audio.sample_rate, self.__fade_duration)
        loop = fade_out(loop, self.audio.sample_rate, self.__fade_duration)
        return AudioData(loop, self.audio.sample_rate)
