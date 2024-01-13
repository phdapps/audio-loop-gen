from numpy import ndarray

from .util import AudioData, trim_silence
from .loop_strategy import LoopStrategy, TransientAligned, CrossFade, FadeInOut, BeatDetect

class LoopGenerator(object):
    def __init__(self, audio_data: ndarray, sample_rate: int, min_loop_duration: int = 30000):
        self.__audio_data = audio_data
        self.__sample_rate = sample_rate
        self.__min_loop_duration = min_loop_duration
        audio = self.__prepare_data()
        self.__strategies = self.__prepare_strategies(audio)

    def generate(self) -> AudioData:
        loop = None
        for strategy in self.__strategies:
            if strategy.evaluate():
                loop = strategy.create_loop()
                loop = loop
                break

        return loop

    def __prepare_data(self) -> AudioData:
        """ Do any necessary preprocessing.
        """
        self.__audio_data = trim_silence(self.__audio_data, self.__sample_rate, top_db=50)
        return AudioData(self.__audio_data, self.__sample_rate)

    def __prepare_strategies(self, audio: AudioData) -> list[LoopStrategy]:
        """
        Prepare the ordered list of strategies to be used for looping.
        """
        strategies = []
        # Add the strategies here in the order of preference
        strategies.append(BeatDetect(
            audio=audio, min_loop_duration=self.__min_loop_duration))
        strategies.append(TransientAligned(
            audio=audio, min_loop_duration=self.__min_loop_duration))
        strategies.append(CrossFade(
            audio=audio, min_loop_duration=self.__min_loop_duration))
        strategies.append(FadeInOut(
            audio=audio, min_loop_duration=self.__min_loop_duration, fade_duration=1000))
        return strategies
