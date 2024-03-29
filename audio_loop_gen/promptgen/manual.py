import asyncio
from typing_extensions import Callable, Concatenate

from .base import PromptGenerator, PS
from ..util import LoopGenParams

class Manual(PromptGenerator):
    """ Generator that reads user input from the console asynchronously. """
    def __init__(self, use_case:str = None, params_callback: Callable[Concatenate[str, int, PS], LoopGenParams] = None):
        super().__init__(use_case=use_case, params_callback = params_callback)

    async def generate(self, max_count: int = 1, seed: int = -1) -> list[LoopGenParams]:
        loop_gen_params = []

        print(f"Enter parameters for up to {max_count} loops" + " for the use case: \"{self.use_case}\"..." if self.use_case else "...")
        for i in range(max_count):
            print(f"Parameters for loop {i + 1}:")
            prompt = await asyncio.to_thread(input, "Enter prompt: ")
            bpm_input = await asyncio.to_thread(input, "Enter bpm: ")
            try:
                bpm = int(bpm_input)
            except ValueError:
                print("Invalid input for bpm. Please use valid integers! Using default bpm of 60.")
                bpm = 60
            loop_gen_params.append(self.params_callback(prompt, bpm))
            if i < max_count - 1:
                go_on = await asyncio.to_thread(input, "Continue? (y/n): ")
                if go_on.lower() != "y":
                    break

        return loop_gen_params