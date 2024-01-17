import uuid6
import logging
import json
from datetime import datetime as dt

from ..util import AudioData, LoopGenParams

class AudioStore(object):
    """ Stores audio using a list of handlers.
    """
    def __init__(self, handlers: list['AudioHandler']):
        self.__handlers = handlers
        self.__logger = logging.getLogger("global")
        
    def store(self, audio: AudioData, params:LoopGenParams):
        """ Saves the given audio using all configured handlers.
        """
        self.__logger.debug("Storing audio: duration=%ds, stereo=%s, sample_rate=%d", audio.duration, audio.is_stereo, audio.sample_rate)
        for handler in self.__handlers:
            try:
                self.__logger.debug("Storing audio with handler %s", handler)
                handler.handle(audio, params)
            except Exception as e:
                self.__logger.error("Error storing audio with handler %s: %s", handler, e)
 
class AudioHandler(object):      
    """ Base class for audio handlers.
    """         
    def __init__(self, keep_metadata:bool = False):
        self.keep_metadata = keep_metadata
    
    def base_name(self, audio: AudioData):
        """ Returns a file name for the given base name and format.
        """
        uuid_str = str(uuid6.uuid7()) # create uuids sortable by creation time
        return dt.utcnow().strftime(
            f"{uuid_str}_{'stereo' if audio.is_stereo else 'mono'}_{audio.duration}ms")
        
    def metadata(self, params: LoopGenParams, audio: AudioData) -> str:
        """ Returns a dictionary of metadata to store along with the audio.
        """
        data = {
            "params": params.to_dict(),
            "duration": audio.duration,
            "sample_rate": audio.sample_rate,
            "is_stereo": audio.is_stereo
        }
            
        return json.dumps(data)
    
    def handle(self, audio: AudioData, params:LoopGenParams):
        """ The specific logic implementation to handle the given audio.
        """
        raise NotImplementedError()