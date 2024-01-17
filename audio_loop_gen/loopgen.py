from .util import AudioData, LoopGenParams, prune_silence
from .loop_strategy import LoopStrategy, TransientAligned, CrossFade, FadeInOut, BeatDetect

class LoopGenerator(object):
    SILENCE_TOP_DB = 45
    def __init__(self, audio: AudioData, params: LoopGenParams):
        assert audio is not None
        assert params is not None
        self.__min_loop_duration = params.min_duration * 1000
        self.__audio = self.__prepare_data(audio)
        self.__strategies = self.__prepare_strategies()

    def generate(self) -> AudioData:
        loop = None
        for strategy in self.__strategies:
            if strategy.evaluate():
                loop = strategy.create_loop()
                loop = loop
                break

        return loop

    def __prepare_data(self, audio: AudioData) -> AudioData:
        """ Do any necessary preprocessing.
        """
        return prune_silence(audio, top_db=type(self).SILENCE_TOP_DB)

    def __prepare_strategies(self) -> list[LoopStrategy]:
        """
        Prepare the ordered list of strategies to be used for looping.
        """
        strategies = []
        # Add the strategies here in the order of preference
        strategies.append(BeatDetect(
            audio=self.__audio, min_loop_duration=self.__min_loop_duration))
        strategies.append(TransientAligned(
            audio=self.__audio, min_loop_duration=self.__min_loop_duration))
        strategies.append(CrossFade(
            audio=self.__audio, min_loop_duration=self.__min_loop_duration))
        strategies.append(FadeInOut(
            audio=self.__audio, min_loop_duration=self.__min_loop_duration, fade_duration=500))
        return strategies
