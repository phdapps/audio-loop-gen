import logging

from typing import Tuple

from numpy import ndarray
import torch

from audiocraft.models import MusicGen

from .util import set_all_seeds
DEFAULT_MODEL_ID = "facebook/musicgen-stereo-large"


class MusicGenerator:
    def __init__(self, model_id: str = None, device: str = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.__load(model_id)
        self.__logger = logging.getLogger("global")

    def generate(self, prompt: str, bpm: int = 66, seed: int = -1, max_duration: int = 60, top_k: int = 250, top_p: float = 0.0, temperature: float = 1.0, cfg_coef: int = 3) -> Tuple[int, ndarray]:
        if not seed or seed < 0:
            seed = torch.seed() % (2**32 - 1)
        elif seed >= 2**32 - 1:
            raise ValueError(f"Seed must be less than {2**32 - 1}")
        set_all_seeds(seed)

        if bpm < 15 or bpm > 300:
            raise ValueError(
                f"Invalid bpm {bpm}, must be between 15.0 and 300.0")

        prompt = f"{prompt} bpm: {bpm}"

        self.__logger.info(
            "Generating music using prompt \"%(prompt)s\" and seed %(seed)d...", extra={prompt: prompt, seed: seed})

        self.model.set_generation_params(
            duration=max_duration, top_k=top_k, top_p=top_p, temperature=temperature, cfg_coef=cfg_coef)

        wav = self.model.generate([prompt], progress=True)[0].cpu()

        sample_rate = self.model.sample_rate
        return sample_rate, wav.numpy()

    def __load(self, model_id: str = None):
        if model_id is None:
            model_id = DEFAULT_MODEL_ID
        self.model = MusicGen.get_pretrained(model_id, device=self.device)
