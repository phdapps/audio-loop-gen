import os
import tempfile
import logging
import boto3

from .base import AudioHandler
from ..util import AudioData, LoopGenParams, export_audio

class S3DataHandler(AudioHandler):
    """ Uploads the audio to an S3 bucket.
    """
    def __init__(self, bucket: str, prefix: str = None, format: str = None, keep_metadata: bool = False):
        super().__init__(keep_metadata)
        self.__bucket = bucket
        self.__prefix = prefix if prefix else ""
        self.__format = format if format and format != "" else "wav"
        self.__s3_client = boto3.client('s3')
        self.__logger = logging.getLogger("general")
        
    def base_name(self, audio: AudioData):
        """ Returns a file name for the given base name and format.
        """
        return f"{self.__prefix}{super().base_name(audio)}"
    
    def handle(self, audio: AudioData, params:LoopGenParams):
        """ Saves the given audio to a file in the S3 bucket encoded in the correct format and under the configured path (prefix).

        Args:
            audio (AudioData): The audio to save.
        """
        temp_file = None
        # Create a temporary file with the correct extension, and make sure it's closed
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{self.__format}", delete=False) as f:
                temp_file = f
            # export to the temp file
            noext, _ = os.path.splitext(temp_file.name)
            export_audio(audio, noext, self.__format)
            with open(temp_file.name, "rb") as f:
                data = f.read()
            base_name = self.base_name(audio)
            key = f"{base_name}.{self.__format}"
            self.__s3_client.put_object(Bucket=self.__bucket, Key=key, Body=data)
            if self.keep_metadata:
                metadata = self.metadata(params, audio)
                key = f"{base_name}.json"
                self.__s3_client.put_object(Bucket=self.__bucket, Key=key, Body=metadata)
        finally:
            if temp_file:
                try:
                    temp_file.close()
                    os.remove(temp_file.name)
                except Exception as e:
                    self.__logger.error("Error closing/deleting temp file: %s", e)