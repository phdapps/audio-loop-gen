
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

## How to use

The project contains a helper CLI module:

        python run.py --mode loopgen ...