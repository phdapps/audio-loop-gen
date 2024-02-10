import os
from enum import Enum
from typing_extensions import Annotated
import asyncio
import logging

import typer

from .audiogen import AudioGenerator
from .loopgen import LoopGenerator
from .store import AudioHandler, AudioStore, FileDataHandler, S3DataHandler
from .util import AudioData, LoopGenParams, setup_logging
from .promptgen import Ollama, OpenAI, Manual

class PromptProvider(str, Enum):
    manual = "manual"
    openai = "openai"
    ollama = "ollama"
    
class SaveFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"
    ogg = "ogg"
    flac = "flac"
    
def create_store(
    save_format:str,
    dest_path:str,
    file_prefix:str,
    s3_bucket:str,
    s3_path:str,
    keep_metadata:bool) -> AudioStore:
    # need to store somewhere
    if s3_bucket is None and dest_path is None:
        print("No storage destination specified, using current directory!")
        dest_path = "."

    handlers:list[AudioHandler] = []
    if s3_bucket is not None:
        prefix = ""
        if s3_path:
            prefix += s3_path
            if not prefix.endswith("/"):
                prefix += "/"
        if file_prefix:
            prefix += file_prefix
        handlers.append(S3DataHandler(bucket=s3_bucket, prefix=prefix, format=save_format, keep_metadata=keep_metadata))
    if dest_path:
        handlers.append(FileDataHandler(dest=dest_path, prefix=file_prefix, format=save_format, keep_metadata=keep_metadata))
    
    return AudioStore(handlers=handlers)

def create_prompt_provider(prompt_provider:PromptProvider, llm_model:str, use_case:str, max_duration: int, min_duration: int) -> PromptProvider:
    def params_callback(prompt:str, bpm:int, **kwargs) -> LoopGenParams:
        _max_duration = kwargs.pop("max_duration", max_duration)
        _min_duration = kwargs.pop("min_duration", min_duration)
        _min_duration = min(_max_duration, max(10, _min_duration))
        return LoopGenParams(prompt=prompt, bpm=bpm, max_duration=_max_duration,
                            min_duration=_min_duration, **kwargs)
        
    if prompt_provider == PromptProvider.ollama:
        promptgen = Ollama(model_id=llm_model, use_case=use_case, params_callback=params_callback)
    elif prompt_provider == PromptProvider.openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable!")
        promptgen = OpenAI(api_key=api_key, model_id=llm_model, use_case=use_case, params_callback=params_callback)
    else:
        promptgen = Manual(use_case=use_case, params_callback=params_callback)
        
    return promptgen

def create_audio_generator(audio_model:str) -> AudioGenerator:
    print("Loading audio model...")
    audiogen = AudioGenerator(model_id=audio_model)
    def progress_callback(generated, total):
        step = total // 20
        if step > 0 and generated % step == 0:
            print(f'.', end='', flush=True)
    audiogen.set_custom_progress_callback(progress_callback)
    return audiogen

async def auto_loop(prompt_provider:PromptProvider, audio_store: AudioStore, audiogen: AudioGenerator):
    logger = logging.getLogger("global")
    while True:
        try:
            try:
                params_list = await prompt_provider.generate(max_count=10)
            except Exception as ex:
                logger.error("Error while generating prompts: %s", ex, exc_info=True)
                await asyncio.sleep(0.1)
                continue
            for params in params_list:
                while True: # retry the params until a loop is generated
                    try:
                        print(f"Generating music, be patient...")
                        sr, audio_data = audiogen.generate(params)
                        ad = AudioData(audio_data, sr)
                        loopgen = LoopGenerator(ad, params)
                        loop = loopgen.generate()
                        if loop:
                            audio_store.store(loop, params)
                            print("\nSaved")
                            break
                        else:
                            print("\nUnsuitable for looping, retrying...")
                            continue
                    except Exception as e:
                        logger.error("Error while generating loop: %s", e, exc_info=True)
                        break
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            break

cli = typer.Typer()

@cli.command(help="Generate a single audio loop reading all arguments from the command line. No LLM is used for the textual prompt generation.")
def generate(
    # audio generation options
    prompt:Annotated[str, typer.Argument(help="The prompt to use for the audio generation model.")],
    audio_model:Annotated[str, typer.Option(help="The name of the MusicGen model to use.")] = None,
    bpm:Annotated[int, typer.Option(help="The beats per minute to target for the generated audio", min=24, max=240)] = 60,
    max_duration:Annotated[int, typer.Option(help="The maximum duration in seconds of the generated loop", min=8, max=128)] = 66,
    min_duration:Annotated[int, typer.Option(help="The minimum duration in seconds for the generated loop", min=8, max=128)] = 40,
    seed:Annotated[int, typer.Option(help="The seed to use for the varios random generators to allow reproducability. -1 for random.")] = -1,
    # storage options
    save_format:Annotated[SaveFormat, typer.Option(help="The format to use when exporting the generated audio file.")] = SaveFormat.mp3,
    dest_path:Annotated[str, typer.Option(help="Local path where to save the generated audio files.")] = None,
    file_prefix:Annotated[str, typer.Option(help="Prefix to use for the generated audio file names.")] = None,
    s3_bucket:Annotated[str, typer.Option(help="S3 bucket name where to save the generated audio files. AWS credentials must be configured in environment variables or in a corresponding credentials file.")] = None,
    s3_path:Annotated[str, typer.Option(help="Bucket path/prefix to save the files to")] = None,
    keep_metadata:Annotated[bool, typer.Option(help="Save a metadata json file with the audio file.", is_flag=True)]=False,
    # logging options
    log_level:Annotated[int, typer.Option(help="Log level as defined in the logging module (i.e. DEBUG=10, INFO=20 etc)")] = None):
    
    setup_logging(log_level=log_level)
    
    audio_store = create_store(save_format.value, dest_path, file_prefix, s3_bucket, s3_path, keep_metadata)
    
    # generate an audio segment using the given parameters
    audiogen = create_audio_generator(audio_model)
    
    min_duration = min(max_duration, max(10, min_duration))
    params = LoopGenParams(prompt=prompt, bpm=bpm, max_duration=max_duration,
                            min_duration=min_duration, seed=seed)
    sr, audio_data = audiogen.generate(params)
    
    # trim it to form a loop
    loopgen = LoopGenerator(AudioData(audio_data, sr), params)
    loop = None
    while not loop:
        loop = loopgen.generate()
        if loop:
            # save it
            audio_store.store(loop, params)
            print("Saved")
        else:
            print("Unsuitable for looping, retrying...")

@cli.command(help="Runs the loop generation in auto mode, using an LLM to generate the prompts and immediately pipe them to the audio generation module, eventually storing the results.")
def auto(
    # prompt generation options
    prompt_provider:Annotated[PromptProvider, typer.Option(help="The provider to use for the prompt generation. 'ollama' expects a locally running Ollama!")] = PromptProvider.openai,
    llm_model:Annotated[str, typer.Option(help="The name of the LLM model to use with the selected provider. If not specified, a default model will be used.")] = None,
    use_case:Annotated[str, typer.Option(help="Extra use case details to send to the prompt generator to influence the type of melodies it would focus on.")] = None,
    # audio generation options
    audio_model:Annotated[str, typer.Option(help="The name of the MusicGen model to use.")] = None,
    max_duration:Annotated[int, typer.Option(help="The maximum duration in seconds of the generated loop", min=8, max=128)] = 66,
    min_duration:Annotated[int, typer.Option(help="The minimum duration in seconds for the generated loop", min=8, max=128)] = 40,
    # storage options
    save_format:Annotated[SaveFormat, typer.Option(help="The format to use when exporting the generated audio file.")] = SaveFormat.mp3,
    dest_path:Annotated[str, typer.Option(help="Local path where to save the generated audio files.")] = None,
    file_prefix:Annotated[str, typer.Option(help="Prefix to use for the generated audio file names.")] = None,
    s3_bucket:Annotated[str, typer.Option(help="S3 bucket name where to save the generated audio files. AWS credentials must be configured in environment variables or in a corresponding credentials file.")] = None,
    s3_path:Annotated[str, typer.Option(help="Bucket path/prefix to save the files to")] = None,
    keep_metadata:Annotated[bool, typer.Option(help="Save a metadata json file with the audio file.", is_flag=True)]=False,
    # logging options
    log_level:Annotated[int, typer.Option(help="Log level as defined in the logging module (i.e. DEBUG=10, INFO=20 etc)")] = None):
    setup_logging(log_level=log_level)
    
    audio_store = create_store(save_format.value, dest_path, file_prefix, s3_bucket, s3_path, keep_metadata)
    
    prompt_provider = create_prompt_provider(prompt_provider, llm_model, use_case, max_duration, min_duration)
    
    audiogen = create_audio_generator(audio_model)
    
    try:
        asyncio.run(auto_loop(prompt_provider, audio_store, audiogen))
    except asyncio.exceptions.CancelledError:
        print("Exiting...")