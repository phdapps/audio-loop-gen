
import numpy as np
from numpy import ndarray
import pydub
    
def ndarray_to_audio_segment(audio_data:ndarray, sample_rate:int) -> pydub.AudioSegment:
    """Converts a NumPy array to an AudioSegment.

    Args:
        audio_data (ndarray): The NumPy array containing the audio data.
        sample_rate (int, optional): The sample rate of the data. Defaults to 44100.

    Returns:
        pydub.AudioSegment: The AudioSegment containing the audio data.
    """
    audio_data = (audio_data * 32767).astype(np.int16)
    is_stereo = audio_data.ndim == 2 and audio_data.shape[0] == 2
    if is_stereo:
        # Interleave the 2 stereo channels (left and right) if the audio is stereo
        # Stacks the 2 channels as 2 rows in a 2D array
        # then flattens the array "Fortran-style" which is supposed to interleave the 2 channels by taking each element from the 2 rows alternately
        audio_data = np.vstack((audio_data[0], audio_data[1])).reshape((-1,), order='F')
    
    channels = 2 if is_stereo else 1
    # sample_width is 2 bytes for 16-bit audio
    return pydub.AudioSegment(audio_data.tobytes(), frame_rate=sample_rate, sample_width=2, channels= channels)

def audio_segment_to_ndarray(audio_segment:pydub.AudioSegment) -> ndarray:
    """Converts an AudioSegment to a NumPy array.

    Args:
        audio_segment (pydub.AudioSegment): The AudioSegment to be converted.

    Returns:
        ndarray: The NumPy array containing the audio data.
    """
    audio_data:ndarray = np.frombuffer(audio_segment.raw_data, dtype=np.int16)
    channels = audio_segment.channels
    if channels == 2:
        # Deinterleave the 2 stereo channels (left and right) if the audio is stereo
        # Reshape the array into 2 rows, then transpose it to deinterleave the 2 channels by taking each element from the 2 rows alternately
        audio_data = audio_data.reshape((2, -1)).T
    return audio_data

def crossfade(audio_segment: pydub.AudioSegment, fade_duration: int) -> pydub.AudioSegment:
    """Crossfades the loop with itself to create a seamless loop.

    Args:
        audio_segment (pydub.AudioSegment): The audio data for the loop.
        fade_duration (int, optional): The duration of the crossfade in milliseconds. Defaults to 1000.

    Returns:
        ndarray: The crossfaded loop.
    """
    # Ensure fade_duration is not longer than 1/3 of the audio file's duration
    fade_duration = min(fade_duration, len(audio_segment) // 3)
    
    # Create a crossfade loop
    start_segment = audio_segment[:fade_duration]
    end_segment = audio_segment[-fade_duration:]
    overlay = end_segment.overlay(start_segment, gain_during_overlay=-6)
    
    audio_segment = audio_segment[0, -fade_duration] + overlay

    return audio_segment