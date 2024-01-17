import logging

from typing import Tuple, Callable

from numpy import ndarray
import torch

from audiocraft.models import MusicGen

from .util import set_all_seeds, AudioGenParams
DEFAULT_MODEL_ID = "facebook/musicgen-stereo-large"

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

        self.__model.set_generation_params(
            duration=params.max_duration, top_k=params.top_k, top_p=params.top_p, temperature=params.temperature, cfg_coef=params.cfg_coef, extend_stride=12)

        wav = self.__model.generate([prompt], progress=self.__progress)[0].cpu()

        sample_rate = self.__model.sample_rate
        return sample_rate, wav.numpy()
    
    def set_custom_progress_callback(self, callback:Callable[[int, int],None]):
        self.__model.set_custom_progress_callback(callback)

    def __load(self, model_id: str = None):
        if model_id is None:
            model_id = DEFAULT_MODEL_ID
        self.__model = MusicGen.get_pretrained(model_id, device=self.device)
