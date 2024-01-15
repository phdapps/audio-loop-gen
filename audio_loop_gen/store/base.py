import uuid6
import logging
from datetime import datetime as dt

from ..util import AudioData

class AudioStore(object):
    """ Stores audio using a list of handlers.
    """
    def __init__(self, handlers: list['AudioHandler']):
        self.__handlers = handlers
        self.__logger = logging.getLogger("global")
        
    def store(self, audio: AudioData):
        """ Saves the given audio using all configured handlers.
        """
        self.__logger.debug("Storing audio: duration=%ds, stereo=%s, sample_rate=%d", audio.duration, audio.is_stereo, audio.sample_rate)
        for handler in self.__handlers:
            try:
                self.__logger.debug("Storing audio with handler %s", handler)
                handler.handle(audio)
            except Exception as e:
                self.__logger.error("Error storing audio with handler %s: %s", handler, e)
                
    def base_name(self, audio: AudioData):
        """ Returns a file name for the given base name and format.
        """
        uuid_str = str(uuid6.uuid7()) # create uuids sortable by creation time
        return dt.utcnow().strftime(
            f"{uuid_str}_{'stereo' if audio.is_stereo else 'mono'}_{audio.duration}ms")
 
class AudioHandler(object):      
    """ Base class for audio handlers.
    """         
    def handle(self, audio: AudioData):
        """ The specific logic implementation to handle the given audio.
        """
        raise NotImplementedError()