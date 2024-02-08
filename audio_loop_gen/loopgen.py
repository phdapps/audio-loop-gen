from .util import AudioData, LoopGenParams, prune_silence, equal_power_crossfade
from .loop_strategy import LoopStrategy, TransientAligned, CrossFade, BeatDetect
# from .loop_strategy import FadeInOut

CROSSFADE_DURATION_MS = 1000

class LoopGenerator(object):
    SILENCE_TOP_DB = 45

    def __init__(self, audio: AudioData, params: LoopGenParams, crossfade: bool = True):
        assert audio is not None
        assert params is not None
        self.__params = params
        self.__min_loop_duration = params.min_duration * 1000
        self.__audio = self.__prepare_data(audio)
        self.__strategies = self.__prepare_strategies()
        self.__crossfade = crossfade

    def generate(self) -> AudioData:
        loop = None
        for strategy in self.__strategies:
            if strategy.evaluate():
                loop = strategy.create_loop()
                loop = loop
                self.__params.strategy_id = strategy.strategy_id
                break

        if self.__crossfade  and loop is not None:
            wav = equal_power_crossfade(loop.audio_data, loop.sample_rate, CROSSFADE_DURATION_MS, min_level=0.25, max_level=0.75)
            loop = AudioData(wav, loop.sample_rate)
        return loop

    def __prepare_data(self, audio: AudioData) -> AudioData:
        """ Do any necessary preprocessing.
        """
        return prune_silence(audio, top_db=type(self).SILENCE_TOP_DB)

    def __prepare_strategies(self) -> list[LoopStrategy]:
        """
        Prepare the ordered list of strategies to be used for looping.
        """
        available_strategy_types: list[type] = [
            BeatDetect, TransientAligned, CrossFade]
        strategies: [LoopStrategy] = []
        for stype in available_strategy_types:
            if (not self.__params.strategy_id) or stype.strategy_id == self.__params.strategy_id:
                strategies.append(stype(
                    audio=self.__audio, min_loop_duration=self.__min_loop_duration))
        # Doesn't produce very good/seamless results, but could be enabled as a fallback if ignoring a generated audio is not an option
        # if not self.__params.strategy_id or FadeInOut.strategy_id == self.__params.strategy_id:
        #     strategies.append(FadeInOut(audio=self.__audio, min_loop_duration=self.__min_loop_duration, fade_duration=500))
        return strategies
