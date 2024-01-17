import os
from .base import AudioHandler

from ..util import AudioData, LoopGenParams, export_audio
class FileDataHandler(AudioHandler):
    """ Stores the audio to a local file.
    """
    
    def __init__(self, dest: str = None, prefix:str = None, format:str=None, keep_metadata: bool = False):
        """ Creates a new FileDataHandler.

        Args:
            dest (str, optional): The destination folder where to store files. Defaults to the current folder.
            prefix (str, optional): A prefix to use in the saved filename. Defaults to empty.
            format (str, optional): The audio format to encode the file in. Defaults to "wav".
        """
        super().__init__(keep_metadata)
        self.__dest = dest if dest and dest != "" else "."
        self.__prefix = prefix if prefix else ""
        self.__format = format if format and format != "" else "mp3"
        
    def base_name(self, audio: AudioData):
        """ Returns a file name for the given base name and format.
        """
        return f"{self.__prefix}{super().base_name(audio)}"
    
    def handle(self, audio: AudioData, params:LoopGenParams):
        """ Saves the given audio to a file encoded in the correct format.

        Args:
            audio (AudioData): The audio to save.
        """
        file_path = os.path.join(self.__dest, self.base_name(audio)) # no extension!
        export_audio(audio, file_path, self.__format)
        if self.keep_metadata:
            metadata = self.metadata(params, audio)
            with open(file_path + ".json", "w") as f:
                f.write(metadata)