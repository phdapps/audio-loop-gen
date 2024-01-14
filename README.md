
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

- To generate prompts locally using Mistral 7b you'll need *ollama*. [Installation instructions](https://github.com/jmorganca/ollama)
  
  - Works out-of-the box on Linux and Mac OS
  
  - Requires CUDA-enabled WSL2 for Windows or Docker (for GPU support you'll still need the WSL2-based backend) [Start here](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)

  - In a new terminal start the server and leave it running:
  
        ollama serve

  - In yet another terminal download the Mistral 7B model (only the very fisrt time!):

        ollama pull mistral

## Using the CLI module

The project contains a helper CLI module. Usage:

        run.py [--mode {loopgen,promptgen}] [--help] [... mode options ...]

                options:
                --mode {loopgen,promptgen}
                --help


                [mode = loopgen]
                Generates audio loops from the CLI arguments and/or requests received from websocket clients connected to the port where the internal server is listening.
                Audio from the CLI arguments is saved to a file, while the server sends it as a response to the client. 
                The server also sends periodic progress update messages to the client as the generation is happening.

                mode options:
                --model MODEL                   Model ID to use for the generation. If not specified, the default model will be used.
                --prompt PROMPT                 The prompt to use for the audio generation model.
                --bpm BPM                       Beats per minute (bpm) to target in the generated audio from the --prompt argument.
                --max_duration MAX_DURATION     Maximum duration in seconds of the generated audio from the --prompt argument. The loop processing can reduce the final duration significantly!
                --dest_path DEST_PATH           Path where to save the generated audio from the --prompt argument.
                --file_name FILE_NAME           File name to save the generated audio from the --prompt argument. If not specified, a name will be generated based on the current date and time.
                --seed SEED                     Seed to use for the generated audio from the --prompt argument. If not specified, a random seed will be used.
                --listen LISTEN                 A websocket port to listen to for generation commads and send back progress updates and the generated audio data.
                --log_level LOG_LEVEL           Log level as defined in the logging module (i.e. DEBUG=10, INFO=20 etc).

                [mode = promptgen]
                Uses an LLM model to generate prompts and other settings for loop generation and sends them to a loopgen server then stores the resulting audio loops.

                options:
                --provider {llama,openai}       The LLM provider to use for the prompt generation. 'llama' requires running it locally!
                --model MODEL                   The name of the LLM model to use. Default for llama is 'mistral' and default for openai is 'gpt-3.5-turbo-1106'.
                --host HOST                     Host to connect to for sending the prompts.
                --port PORT                     Host to connect to for sending the prompts.
                --save_path SAVE_PATH           Local path where to save the audio files received from the server.
                --save_s3 SAVE_S3               S3 bucket name where to save the audio files received from the server. AWS credentials must be configured in environment variables or in a corresponding credentials file.
                --save_prefix SAVE_PREFIX       Prefix for the saved filenames (could start with a relative path).
