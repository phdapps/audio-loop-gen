import os
import asyncio
from .promptgen import Ollama, OpenAI, Manual

def main():
    print("OPEN_AI:")
    api_key = os.environ.get("OPENAI_API_KEY")
    generator = OpenAI(api_key=api_key)
    params = asyncio.run(generator.generate(max_count=10))
    print([str(p) for p in params])