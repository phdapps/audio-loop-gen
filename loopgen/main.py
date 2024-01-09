import os.path as path
from argparse import ArgumentParser
from datetime import datetime as dt

from .generator import MusicGenerator
from .loopgen import LoopPipeline

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dest_path', type=str, default=".")
    parser.add_argument('--bpm', type=int, default=66)
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--file_name', type=str, default="")
    parser.add_argument('--max_duration', type=int, default=66) 
    return parser.parse_args()
    
def main():
    args = parse_args()
    generator = MusicGenerator()
    sr, audio_data = generator.generate(prompt=args.prompt, bpm=args.bpm, max_duration=args.max_duration)
    loopgen = LoopPipeline(audio_data, sr)
    loop = loopgen.execute()
    file_name = args.file_name if args.file_name else dt.utcnow().strftime(f"%Y%m%d%H%M%S%f-{args.bpm}bpm.mp3")
    loop.export(path.join(args.dest_path, args.file_name), format="mp3")
    
if __name__ == "__main__":
    main()