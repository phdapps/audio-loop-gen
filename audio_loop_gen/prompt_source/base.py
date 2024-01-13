class PromptSource(object):
    def __init__(self) -> None:
        pass
    
    def generate(self) -> list[str]:
        raise NotImplementedError
    

class AudioGenPrompt(object):
    def __init__(self, data: str):
        self.__parse(data)
        
    def __parse(self, data: str):
        parts = data.split("|") 
        self.prompt = parts[0]
        self.bpm = int(parts[1])