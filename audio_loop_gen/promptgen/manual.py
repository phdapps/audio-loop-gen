import asyncio

from .base import PromptGenerator
from ..util import LoopGenParams

class Manual(PromptGenerator):
    """ Generator that reads user input from the console asynchronously. """

    async def generate(self, max_count: int = 1) -> list[LoopGenParams]:
        loop_gen_params = []

        print(f"Enter parameters for up to {max_count} loops...")
        for i in range(max_count):
            print(f"Parameters for loop {i + 1}:")
            prompt = await asyncio.to_thread(input, "Enter prompt: ")
            bpm_input = await asyncio.to_thread(input, "Enter bpm: ")
            try:
                bpm = int(bpm_input)
            except ValueError:
                print("Invalid input for bpm. Please use valid integers! Using default bpm of 60.")
                bpm = 60
            loop_gen_params.append(LoopGenParams(prompt=prompt, bpm=bpm))
            if i < max_count - 1:
                go_on = await asyncio.to_thread(input, "Continue? (y/n): ")
                if go_on.lower() != "y":
                    break

        return loop_gen_params