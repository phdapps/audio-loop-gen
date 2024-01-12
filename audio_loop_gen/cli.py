#!/usr/bin/env python

import os.path as path
from argparse import ArgumentParser
from datetime import datetime as dt

from audio_loop_gen.generator import MusicGenerator
from audio_loop_gen.looper import Looper
from audio_loop_gen.util import export_audio, setup_global_logging

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--bpm', type=int, default=66)
    parser.add_argument('--max_duration', type=int, default=66)
    parser.add_argument('--dest_path', type=str, default=".")
    parser.add_argument('--file_name', type=str, default="")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--log_level', type=int, default=-1, help="Log level as defined in the logging module (i.e. DEBUG=10, INFO=20 etc).")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_global_logging(log_level=args.log_level)
    generator = MusicGenerator(model_id=args.model)
    sr, audio_data = generator.generate(
        prompt=args.prompt, bpm=args.bpm, max_duration=args.max_duration, seed = args.seed)
    # don't loose more than 1/3 of the audio
    looper = Looper(audio_data, sr, min_loop_duration=(
        args.max_duration*1000)//2)
    loop = looper.execute()
    file_name = args.file_name if args.file_name else dt.utcnow().strftime(
        f"%Y%m%d%H%M%S%f-{args.max_duration}s-{args.bpm}bpm")
    export_audio(loop, path.join(args.dest_path, file_name), format="mp3")
