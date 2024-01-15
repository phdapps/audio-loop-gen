#!/usr/bin/env python

import sys
import os.path as path
from argparse import ArgumentParser
from datetime import datetime as dt
from io import StringIO

from audio_loop_gen.audiogen import AudioGenerator
from audio_loop_gen.loopgen import LoopGenerator
from audio_loop_gen.util import export_audio, AudioData, LoopGenParams
from audio_loop_gen.logging import setup_global_logging
from audio_loop_gen.server import LoopGeneratorServer


def loopgen_args_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default=None,
                        help="Model ID to use for the generation. If not specified, the default model will be used.")
    parser.add_argument('--prompt', type=str,
                        default="The prompt to use for the audio generation model.")
    parser.add_argument('--bpm', type=int, default=60,
                        help="Beats per minute (bpm) to target in the generated audio from the --prompt argument.")
    parser.add_argument('--max_duration', type=int, default=33,
                        help="Maximum duration in seconds of the generated audio from the --prompt argument. The loop processing can reduce the final duration significantly!")
    parser.add_argument('--dest_path', type=str, default=".",
                        help="Path where to save the generated audio from the --prompt argument.")
    parser.add_argument('--file_name', type=str, default="",
                        help="File name to save the generated audio from the --prompt argument. If not specified, a name will be generated based on the current date and time.")
    parser.add_argument('--seed', type=int, default=-1,
                        help="Seed to use for the generated audio from the --prompt argument. If not specified, a random seed will be used.")
    parser.add_argument('--listen', type=int, default=0,
                        help="A websocket port to listen to for generation commads and send back progress updates and the generated audio data.")
    parser.add_argument("--mode", type=str, help="loopgen (optional)")
    return parser


def do_loopgen(argv: list[str]):
    parser = loopgen_args_parser()
    args = parser.parse_args(argv)
    if (not args.prompt) and (not args.listen or args.listen < 1):
        raise ValueError("Either --prompt or --listen must be specified.")

    audiogen = AudioGenerator(model_id=args.model)

    if args.prompt:
        # Generate the CLI prompt audio first before listening for more prompts
        params = LoopGenParams(prompt=args.prompt, bpm=args.bpm, max_duration=args.max_duration,
                               min_duration=args.max_duration//2, seed=args.seed)
        sr, audio_data = audiogen.generate(params)
        # don't loose more than 1/3 of the audio
        loopgen = LoopGenerator(AudioData(audio_data, sr), params)
        loop = loopgen.generate()
        file_name = args.file_name if args.file_name else dt.utcnow().strftime(
            f"%Y%m%d%H%M%S%f-{args.max_duration}s-{args.bpm}bpm")
        export_audio(loop, path.join(args.dest_path, file_name), format="wav")

    if args.listen and args.listen > 0:
        # Start the websocket server
        server = LoopGeneratorServer(audiogen, port=args.listen)
        server.start()


def promptgen_args_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--provider', type=str,
                        choices=["ollama", "openai"], default="openai", help="The LLM provider to use for the prompt generation. 'ollama' requires running it locally!")
    parser.add_argument("--model", type=str, default=None,
                        help="The name of the LLM model to use with the selected provider. If not specified, a default model will be used.")
    parser.add_argument('--host', type=str, default="localhost",
                        help="Host to connect to for sending the prompts.")
    parser.add_argument('--port', type=str, default="localhost",
                        help="Host to connect to for sending the prompts.")
    parser.add_argument('--save_path', type=str, default=".",
                        help="Local path where to save the audio files received from the server.")
    parser.add_argument('--save_s3', type=str, default=None,
                        help="S3 bucket name where to save the audio files received from the server. AWS credentials must be configured in environment variables or in a corresponding credentials file.")
    parser.add_argument('--save_prefix', type=str, default="",
                        help="Prefix for the saved filenames. For S3, it can start with a path prefix.")
    parser.add_argument("--mode", type=str, help="promptgen")
    return parser


def do_promptgen(argv: list[str]):
    parser = promptgen_args_parser()
    args = parser.parse_args(argv)

    pass


def get_parser_help(parser: ArgumentParser) -> str:
    old_stdout = sys.stdout
    sys.stdout = temp_stdout = StringIO()
    parser.print_help()
    sys.stdout = old_stdout
    return temp_stdout.getvalue()


def combine_help_messages(mode_parser: ArgumentParser, loopgen_parser: ArgumentParser, promptgen_parser: ArgumentParser) -> str:
    result = get_parser_help(mode_parser)
    result += "\n\n[mode = loopgen]\n"
    result += get_parser_help(loopgen_parser)
    result += "\n\n[mode = promptgen]\n"
    result += get_parser_help(promptgen_parser)
    result += "\n"
    return result


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--mode", type=str,
                        choices=["loopgen", "promptgen"], default="loopgen")
    parser.add_argument("--help", action="store_true")
    parser.add_argument('--log_level', type=int, default=-1,
                        help="Log level as defined in the logging module (i.e. DEBUG=10, INFO=20 etc).")
    args, remaining_argv = parser.parse_known_args()
    if args.help:
        help = combine_help_messages(
            parser, loopgen_args_parser(), promptgen_args_parser())
        print(help)
        return

    setup_global_logging(log_level=args.log_level)

    if args.mode == "loopgen":
        do_loopgen(argv=remaining_argv)
    elif args.mode == "promptgen":
        do_promptgen(argv=remaining_argv)
