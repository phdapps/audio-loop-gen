import logging

from typing import Tuple, Callable

import numpy as np
from numpy import ndarray
import torch

from audiocraft.models import MusicGen

from .util import set_all_seeds, AudioGenParams, equal_power_crossfade_arrays
DEFAULT_MODEL_ID = "facebook/musicgen-stereo-large"
MAX_DURATION = 30.0
CONTINUATION_WINDOW_DURATION = 15.0
CONTINUATION_CROSSFADE_DURATION_MS = 500
class AudioGenerator:
    """ Generates an audio segment using one of Meta's Audiocraft:MusicGen models.
    """

    def __init__(self, model_id: str = None, device: str = None, progress:bool=True):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.__progress = progress
        self.__model = None
        self.__load(model_id)
        self.__logger = logging.getLogger("global")

    def generate(self, params: AudioGenParams) -> Tuple[int, ndarray]:
        """ Generates an audio segment using the given parameters.

        Args:
            params (AudioGenParams): The parameters to use for generation.

        Raises:
            ValueError: If the parameters are invalid.

        Returns:
            Tuple[int, ndarray]: The sample rate and audio data.
        """
        if not params:
            raise ValueError("Missing generation params")
        
        seed = params.seed
        if not seed or seed < 0:
            seed = torch.seed() % (2**32 - 1)
        elif seed >= 2**32 - 1:
            raise ValueError(f"Seed must be less than {2**32 - 1}")
        set_all_seeds(seed)

        if params.bpm < 15 or params.bpm > 300:
            raise ValueError(
                f"Invalid bpm {params.bpm}, must be between 15.0 and 300.0")

        prompt = f"{params.prompt} bpm: {params.bpm}"

        self.__logger.info(
            "Generating music using prompt \"%s\" and seed %d...", prompt, seed)
        
        target_duration = params.max_duration

        duration: float = min(target_duration, MAX_DURATION)
        self.__model.set_generation_params(
                    duration=duration, top_k=params.top_k, top_p=params.top_p, temperature=params.temperature, cfg_coef=params.cfg_coef, extend_stride=12)
        
        wav = self.__model.generate([prompt], progress=self.__progress)
        sample_rate = self.__model.sample_rate
        
        result = wav[0].cpu().numpy()
        # If the requested duration is greater than MAX_DURATION, generate the rest using continuations and merge them with crossfades
        if target_duration > duration:
            remaining = target_duration - duration
            while remaining > 0:
                # the generated duration must be longer than the prompt window by at least 1 second or MusicGen raises assertion errors!
                duration = max(min(remaining, MAX_DURATION), CONTINUATION_WINDOW_DURATION + 1.0)
                self.__model.set_generation_params(duration=duration, cfg_coef=6.0) # make it follow the prompt more closely
                prompt_wav = wav[..., -int(CONTINUATION_WINDOW_DURATION*sample_rate):]
                wav = self.__model.generate_continuation(prompt_wav, sample_rate, descriptions=[prompt], progress=self.__progress, return_tokens=False)
                result = equal_power_crossfade_arrays(result, wav[0].cpu().numpy(), sample_rate, CONTINUATION_CROSSFADE_DURATION_MS)
                if remaining < duration:
                    break
                else:
                    remaining -= duration
        total_len = result.shape[-1]
        target_len = int(target_duration*sample_rate)
        if total_len > target_len:
            result = result[..., :target_len]
        return sample_rate, result

    def set_custom_progress_callback(self, callback:Callable[[int, int],None]):
        self.__model.set_custom_progress_callback(callback)

    def __load(self, model_id: str = None):
        if model_id is None:
            model_id = DEFAULT_MODEL_ID
        self.__model = MusicGen.get_pretrained(model_id, device=self.device)
