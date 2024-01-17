
# "audio_loop_gen"

__audio_loop_gen__ is an ML-based, fully automated, audio loop generation pipeline.
It combines ML text and music generation with signal processing to implement an unlimited source of varied loopable music.

## Installation

- Install *ffmpeg* for your OS
  
- Tested with Python 3.11 only.
  
  - Create a dedicated environment with Python 3.11. Using [Anaconda](https://www.anaconda.com/download/) is highly recommended!

        conda create --name loopgen python=3.11

- Clone the code
  
        git clone https://github.com/phdapps/audio-loop-gen

- Install the requirements:

        cd audio-loop-gen
        pip install -r requirements.txt

- To generate prompts locally using something like "Mistral 7b" you'll need *ollama*. [Installation instructions](https://github.com/jmorganca/ollama)
  
  - Works out-of-the box on Linux and Mac OS
  
  - Requires CUDA-enabled WSL2 for Windows or Docker (for GPU support you'll still need the WSL2-based backend) [Start here](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)

  - In a new terminal start the server and leave it running:
  
        ollama serve

  - In yet another terminal download the Mistral 7B model (only the very fisrt time!):

        ollama pull mistral

## Using the CLI module

The project contains a helper CLI module. For list of available commands and general help:

        python run.py --help

Then for help with specific command:

        python run.py auto --help