
# "audio_loop_gen"

__audio_loop_gen__ is an ML-based, fully automated, audio loop generation pipeline.
It combines ML text and music generation with signal processing to implement an unlimited source of varied loopable music.

__audio_loop_gen__ is based on Meta's [MusicGen](https://audiocraft.metademolab.com/) with the following improvements:

- Automated music prompt generation using a secondary LLM such as OpenAI or a locally running open-source model such as "Mistral 7b"

- Audio processing algorithms that clip the generated audio into loopable music, i.e. a melody that can play in a seamless loop

[Examples](https://phdapps.github.io/audio-loop-gen/examples/demo/)

## Installation

- Install *ffmpeg* for your OS

- Install PortAudio for you platform:
  
        sudo apt-get install portaudio19-dev (Linux)

        https://winget.run/pkg/intxcc/pyaudio (Windows)
  
- The library uses [BeatNet](https://github.com/mjhydri/BeatNet) which only works with Python 3.9 or earlier! This code was tested with Python 3.9 only.
  
- Create a dedicated environment with Python 3.9. Using [Anaconda](https://www.anaconda.com/download/) is highly recommended!

        conda create --name loopgen python=3.9

        conda activate loopgen

- Clone the code
  
        git clone https://github.com/phdapps/audio-loop-gen

- Install the requirements:

        cd audio-loop-gen

        pip install -r build-requirements.txt
        
        pip install -r requirements.txt

- To generate prompts locally using something like "Mistral 7b" you'll need *ollama*. [Installation instructions](https://github.com/jmorganca/ollama)
  
  - Works out-of-the box on Linux and Mac OS
  
  - Requires CUDA-enabled WSL2 for Windows or Docker (for GPU support you'll still need the WSL2-based backend) [Start here](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)

  - In a new terminal start the server and leave it running:
  
        ollama serve

  - In yet another terminal download the LLM model you'd like to use (only the very fisrt time!). Mistral is used by default, so:

        ollama pull mistral

## Using the CLI module

The project contains a helper CLI module. For list of available commands and general help:

        python run.py --help

Then for help with specific command:

        python run.py COMMAND --help

Examples:

- Generate a loop from your own prompt:

        python run.py generate "A nostalgic and dreamy melody that takes you back to simpler times with piano and acoustic guitar. Slow and mellow with a steady rhythm." --bpm 60 --max-duration 60 --min-duration 40 --dest-path generated --keep-metadata

  Full usage:

        python run.py generate --help

        Usage: run.py generate [OPTIONS] PROMPT

        Generate a single audio loop reading all arguments from the command line. No LLM is used for the textual prompt generation.

        prompt      TEXT  The prompt to use for the audio generation model. [default: None] [required]

        Options:

        --audio-model                            TEXT                        The name of the MusicGen model to use. [default: facebook/musicgen-stereo-large]

        --bpm                                    INTEGER RANGE [24<=x<=240]  The beats per minute to target for the generated audio [default: 60]

        --max-duration                           INTEGER RANGE [8<=x<=128]   The maximum duration in seconds of the generated loop [default: 66]

        --min-duration                           INTEGER RANGE [8<=x<=128]   The minimum duration in seconds for the generated loop [default: 40]
        
        --seed                                   INTEGER                     The seed to use for the varios random generators to allow reproducability. -1 for random. [default: -1]

        --save-format                            [flac|mp3|ogg|wav]          The format to use when exporting the generated audio file. [default: mp3]
        
        --dest-path                              TEXT                        Local path where to save the generated audio files. [default: None]
        
        --file-prefix                            TEXT                        Prefix to use for the generated audio file names. [default: None]

        --s3-bucket                              TEXT                        S3 bucket name where to save the generated audio files. AWS credentials must be configured in environment variables or in a corresponding credentials file. [default: None]
        
        --s3-path                                TEXT                        Bucket path/prefix to save the files to [default: None]
        
        --keep-metadata    --no-keep-metadata                                Save a metadata json file with the audio file. [default: False]
        
        --log-level                              INTEGER                     Log level as defined in the logging module (i.e. DEBUG=10, INFO=20 etc) [default: 20, i.e info]
        
        --help                                                               Show this message and exit.

- Run an automated pipeline that generates audio loops in bulk
  
        OPENAI_API_KEY=sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ python run.py auto --prompt-provider=openai --keep-metadata --use-case "Background music for levels in an RPG." --dest-path game-music --min-duration 60 --max-duration 80 --keep-metadata

  Full usage:

        python run.py auto --help

        Usage: run.py auto [OPTIONS]

        Runs the loop generation in auto mode, using an LLM to generate the prompts and immediately pipe them to the audio generation module, eventually storing the results.

        Options:

         --prompt-provider                          [manual|openai|ollama]     The provider to use for the prompt generation. 'ollama' expects a locally running Ollama! [default: openai]
         
         --llm-model                                TEXT                       The name of the LLM model to use with the selected provider. If not specified, a default model will be used. [default: None]
         
         --use-case                                 TEXT                       Extra use case details to send to the prompt generator to influence the type of melodies it would focus on. [default: None]
         
         --audio-model                              TEXT                       The name of the MusicGen model to use. [default: facebook/musicgen-stereo-large]
         
         --max-duration                             INTEGER RANGE [8<=x<=128]  The maximum duration in seconds of the generated loop [default: 66]
         
         --min-duration                             INTEGER RANGE [8<=x<=128]  The minimum duration in seconds for the generated loop [default: 40]

         --save-format                              [flac|mp3|ogg|wav]         The format to use when exporting the generated audio file. [default: mp3]
         
         --dest-path                                TEXT                       Local path where to save the generated audio files. [default: None]
         
         --file-prefix                              TEXT                       Prefix to use for the generated audio file names. [default: None]
         
         --s3-bucket                                TEXT                       S3 bucket name where to save the generated audio files. AWS credentials must be configured in environment variables or in a corresponding credentials file. [default: None]
         
         --s3-path                                  TEXT                       Bucket path/prefix to save the files to [default: None] 
         
         --keep-metadata      --no-keep-metadata                               Save a metadata json file with the audio file. [default: no-keep-metadata]
         
         --log-level                                INTEGER                    Log level as defined in the logging module (i.e. DEBUG=10, INFO=20 etc) [default: 20, i.e. INFO]
         
         --help                                                                Show this message and exit.

Notes:

  1. Only tested on Windows and Linux with Nvidia GPUs. I'm not a Mac user and support for Mac OS and Apple hardware is not a priority! Renting a cloud server with a decent GPU is always an option.

  2. When using S3 for storage, the corresponding AWS credentials with write access to the bucket should already be configured.

  3. When using OpenAI for prompt generation the api key should already be set in the OPENAI_API_KEY env variable. Conversely, Ollama should already be running if using it for generating prompts locally.

  4. Use the "--use-case" option to give the LLM some extra hints about the music you want to generate
