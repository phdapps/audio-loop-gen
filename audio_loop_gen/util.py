""" Utility and helper functions for audio processing and loop generation.
"""
import hashlib
import os
import random
from typing import Tuple
import logging
from logging.handlers import RotatingFileHandler
import struct

import numpy as np
from numpy import ndarray
import torch

import librosa

from audiocraft.data.audio import audio_write

class AudioData:
    """ Generic audio data container/wrapper used throughout the library.
    """

    def __init__(self, audio_data: ndarray, sample_rate: int):
        self.__audio_data = audio_data
        self.__sample_rate = sample_rate
        self.__mono_audio_data: ndarray = None
        self.__is_stereo = audio_data.ndim == 2 and audio_data.shape[0] == 2
        if self.__is_stereo:
            self.__length: int = audio_data.shape[1]
        else:
            self.__length: int = len(audio_data)
        self.__duration: int = (
            self.__length * 1000) // sample_rate  # in milliseconds

    @property
    def audio_data(self) -> ndarray:
        return self.__audio_data

    @property
    def sample_rate(self) -> int:
        return self.__sample_rate

    @property
    def mono_audio_data(self) -> ndarray:
        if self.__mono_audio_data is None:
            if self.is_stereo:
                self.__mono_audio_data = np.atleast_2d(self.audio_data.mean(axis=0))
            else:
                self.__mono_audio_data = self.audio_data
        return self.__mono_audio_data

    @property
    def duration(self):
        return self.__duration

    @property
    def length(self):
        return self.__length

    @property
    def is_stereo(self):
        return self.__is_stereo
    
    @staticmethod
    def serialize(audio: 'AudioData') -> bytes:
        num_channels = 2 if audio.is_stereo else 1

        # Pack sample_rate and num_channels as integers (4 bytes each for int32)
        header = struct.pack('ii', audio.sample_rate, num_channels)

        # Convert audio_data to bytes
        audio_data_bytes = audio.audio_data.tobytes()

        # Concatenate the header and audio_data_bytes
        return header + audio_data_bytes
        
    @staticmethod
    def deserialize(data:bytes) -> 'AudioData':
        # Unpack sample_rate and num_channels (first 8 bytes, 4 bytes each for int32)
        sample_rate, num_channels = struct.unpack('ii', data[:8])
        # Extract the audio_data bytes
        audio_data_bytes = data[8:]

        # Reconstruct the audio_data ndarray
        # The dtype is assumed to be float32, and shape depends on num_channels
        if num_channels == 1:
            audio_data = np.frombuffer(audio_data_bytes, dtype=np.float32)
        elif num_channels == 2:
            num_samples_per_channel = len(audio_data_bytes) // (num_channels * np.dtype(np.float32).itemsize)
            audio_data = np.frombuffer(audio_data_bytes, dtype=np.float32).reshape((2, num_samples_per_channel))
            
        audio_data = audio_data.copy() # writable copy
        return AudioData(audio_data, sample_rate)

class LazyLoggable(object):
    def __init__(self, callable, *args, **kwargs):
        self.__callable = callable
        self.__args = args
        self.__kwargs = kwargs

    def __str__(self):
        return self.__callable(*self.__args, **self.__kwargs)

class AudioGenParams(object):
    def __init__(self, 
                 prompt: str, 
                 max_duration: int = 60,
                 bpm: int = 66, 
                 seed: int = -1, 
                 top_k: int = 250, 
                 top_p: float = 0.0, 
                 temperature: float = 1.0, 
                 cfg_coef: int = 3):
        self.__prompt = prompt
        self.__max_duration = max_duration
        self.__bpm = bpm
        self.__seed = seed
        self.__top_k = top_k
        self.__top_p = top_p
        self.__temperature = temperature
        self.__cfg_coef = cfg_coef

    @property
    def prompt(self) -> str:
        return self.__prompt
    
    @property
    def max_duration(self) -> int:
        return self.__max_duration
    
    @property
    def bpm(self) -> int:
        return self.__bpm
    
    @property
    def seed(self) -> int:
        return self.__seed
    
    @property
    def top_k(self) -> int:
        return self.__top_k
    
    @property
    def top_p(self) -> float:
        return self.__top_p
    
    @property
    def temperature(self) -> float:
        return self.__temperature
    
    @property
    def cfg_coef(self) -> int:
        return self.__cfg_coef
    
    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "max_duration": self.max_duration,
            "bpm": self.bpm,
            "seed": self.seed,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "cfg_coef": self.cfg_coef
        }
    
    def __str__(self) -> str:
        return str(self.to_dict())
    
class LoopGenParams(AudioGenParams):
    def __init__(self, 
                 prompt: str, 
                 min_duration: int = -1,
                 max_duration: int = 60,
                 bpm: int = 66, 
                 seed: int = -1, 
                 top_k: int = 250, 
                 top_p: float = 0.0, 
                 temperature: float = 1.0, 
                 cfg_coef: int = 3,
                 strategy_id:str = None):
        super().__init__(prompt, max_duration, bpm, seed, top_k, top_p, temperature, cfg_coef)
        self.__min_duration = min_duration
        self.strategy_id = strategy_id
        
    @property
    def min_duration(self) -> int:
        return self.__min_duration
    
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["min_duration"] = self.min_duration
        data["strategy_id"] = self.strategy_id
        return data

def crossfade(audio_data: np.ndarray, sample_rate: int, crossfade_duration_ms: int, min_level: float = 0.0, max_level = 1.0) -> np.ndarray:
    """ Applies a crossfade effect to the end of an audio segment.

    The crossfade effect is applied to the last `crossfade_duration_ms` milliseconds of the audio segment.
    It blends a faded out version of the end of the audio segment with the faded in version of its start, both of the same duration (crossfade_duration_ms).

    Args:
        audio_data (np.ndarray): The audio data to crossfade
        sample_rate (int): The sample rate of the audio data.
        crossfade_duration_ms (int): The duration of the crossfade effect in milliseconds.

    Raises:
        ValueError: If the crossfade duration is too long for the audio length.

    Returns:
        np.ndarray: The audio data with the crossfade effect applied.
    """
    crossfade_samples = int(sample_rate * crossfade_duration_ms / 1000)

    if crossfade_samples >= audio_data.shape[1] // 2:
        # Crossfade duration is too long for the audio length
        raise ValueError(
            "Crossfade duration is too long for the length of the audio.")

    # Create linear crossfade curves
    fade_out = np.linspace(max_level, min_level, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(min_level, max_level, crossfade_samples, dtype=np.float32)

    # Apply fade-out to the end segment
    end_faded = audio_data[:, -crossfade_samples:] * fade_out

    # Apply fade-in to the start segment
    start_faded = audio_data[:, :crossfade_samples] * fade_in

    # Blend the crossfade region
    crossfaded_region = end_faded + start_faded

    # Construct the final audio
    final_audio = np.concatenate(
        [audio_data[:, :-crossfade_samples], crossfaded_region], axis=1)

    return final_audio

def equal_power_crossfade(audio_data: np.ndarray, sample_rate: int, crossfade_duration_ms: int, min_level: float = 0.0, max_level = 1.0) -> np.ndarray:
    """ Applies an equal power (constant power) crossfade effect to the end of an audio segment.
    
    The crossfade effect is applied to the last `crossfade_duration_ms` milliseconds of the audio segment.
    It blends a faded out version of the end of the audio segment with the faded in version of its start, both of the same duration (crossfade_duration_ms).

    Args:
        audio_data (np.ndarray): The audio data to crossfade
        sample_rate (int): The sample rate of the audio data.
        crossfade_duration_ms (int): The duration of the crossfade effect in milliseconds.

    Raises:
        ValueError: If the crossfade duration is too long for the audio length.

    Returns:
        np.ndarray: The audio data with the crossfade effect applied.
    """
    crossfade_samples = int(sample_rate * crossfade_duration_ms / 1000)

    if crossfade_samples >= audio_data.shape[1] // 2:
        # Crossfade duration is too long for the audio length
        raise ValueError(
            "Crossfade duration is too long for the length of the audio.")

    # Create equal power crossfade curves using sine and cosine
    t = np.linspace(min_level*np.pi / 2, max_level*np.pi / 2, crossfade_samples, dtype=np.float32)
    fade_out = np.cos(t)  # Decreases in volume
    fade_in = np.sin(t)   # Increases in volume

    # Apply fade-out to the end segment
    end_faded = audio_data[:, -crossfade_samples:] * fade_out

    # Apply fade-in to the start segment
    start_faded = audio_data[:, :crossfade_samples] * fade_in

    # Blend the crossfade region
    crossfaded_region = end_faded + start_faded

    # Construct the final audio
    final_audio = np.concatenate(
        [audio_data[:, :-crossfade_samples], crossfaded_region], axis=1)

    return final_audio

def linear_crossfade_arrays(audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int, crossfade_duration_ms: int, min_level: float = 0.0, max_level: float = 1.0) -> np.ndarray:
    """
    Applies a crossfade effect between the end of one audio segment and the start of another.

    The crossfade effect blends the fade-out version of the end of the first audio segment with the fade-in version of the start of the second segment over the specified duration.

    Args:
        audio_a (np.ndarray): The first audio data to crossfade out.
        audio_b (np.ndarray): The second audio data to crossfade in.
        sample_rate (int): The sample rate of the audio data.
        crossfade_duration_ms (int): The duration of the crossfade effect in milliseconds.
        min_level (float, optional): The minimum volume level (default is 0.0).
        max_level (float, optional): The maximum volume level (default is 1.0).

    Raises:
        ValueError: If the crossfade duration is too long for either audio segment.

    Returns:
        np.ndarray: The combined audio data with the crossfade effect applied.
    """
    crossfade_samples = int(sample_rate * crossfade_duration_ms / 1000)
    
    # Validate compatibility of array shapes
    if audio_a.shape[0] != audio_b.shape[0]:
        raise ValueError("Audio segments must have the same number of channels.")


    # Check if either audio segment is too short for the crossfade duration
    if crossfade_samples >= min(audio_a.shape[1], audio_b.shape[1]):
        raise ValueError("Crossfade duration is too long for the length of one or both audio segments.")

    # Create linear crossfade curves
    fade_out = np.linspace(max_level, min_level, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(min_level, max_level, crossfade_samples, dtype=np.float32)

    # Apply fade-out to the end of the first audio segment
    end_faded = audio_a[:, -crossfade_samples:] * fade_out

    # Apply fade-in to the beginning of the second audio segment
    start_faded = audio_b[:, :crossfade_samples] * fade_in

    # Blend the crossfade region
    crossfaded_region = end_faded + start_faded

    # Concatenate the non-crossfaded start of the first audio, the crossfaded region, and the non-crossfaded end of the second audio
    final_audio = np.concatenate([audio_a[:, :-crossfade_samples], crossfaded_region, audio_b[:, crossfade_samples:]], axis=1)

    return final_audio

def equal_power_crossfade_arrays(audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int, crossfade_duration_ms: int, min_level: float = 0.0, max_level: float = 1.0) -> np.ndarray:
    """
    Applies an equal power (constant power) crossfade effect between the end of one audio segment and the start of another.
    
    Args:
        audio_a (np.ndarray): The first audio data to crossfade out.
        audio_b (np.ndarray): The second audio data to crossfade in.
        sample_rate (int): The sample rate of the audio data.
        crossfade_duration_ms (int): The duration of the crossfade effect in milliseconds.
        min_level (float, optional): The minimum volume level (default is 0.0).
        max_level (float, optional): The maximum volume level (default is 1.0).
        
    Raises:
        ValueError: If the crossfade duration is too long for either audio segment, or the 2 are incompatible.
    
    Returns:
        np.ndarray: The combined audio data with the crossfade effect applied.
    """
    # Validate compatibility of array shapes
    if audio_a.shape[0] != audio_b.shape[0]:
        raise ValueError("Audio segments must have the same number of channels.")

    crossfade_samples = int(sample_rate * crossfade_duration_ms / 1000)

    # Check if either audio segment is too short for the crossfade duration
    if crossfade_samples > min(audio_a.shape[1], audio_b.shape[1]):
        raise ValueError("Crossfade duration is too long for the length of one or both audio segments.")

    # Create equal power crossfade curves using sine and cosine
    half_pi = np.pi / 2
    t = np.linspace(min_level*half_pi, max_level*half_pi, crossfade_samples, dtype=np.float32)
    fade_out = np.cos(t)  # Decreases in volume
    fade_in = np.sin(t)   # Increases in volume

    # Apply fade-out to the end of the first audio segment
    end_faded = audio_a[:, -crossfade_samples:] * fade_out

    # Apply fade-in to the beginning of the second audio segment
    start_faded = audio_b[:, :crossfade_samples] * fade_in

    # Blend the crossfade region
    crossfaded_region = end_faded + start_faded

    # Concatenate the non-crossfaded start of the first audio, the crossfaded region, and the non-crossfaded end of the second audio
    final_audio = np.concatenate([audio_a[:, :-crossfade_samples], crossfaded_region, audio_b[:, crossfade_samples:]], axis=1)

    return final_audio

def fade_in(audio_data: np.ndarray, sample_rate: int, fade_duration_ms: int, min_level: float = 0.0, max_level: float = 1.0, in_place: bool = False) -> np.ndarray:
    """ Applies a fade-in effect to the start of an audio segment.

    Args:
        audio_data (np.ndarray): The audio data to fade in.
        sample_rate (int): The sample rate of the audio data.
        fade_duration_ms (int): The duration of the fade-in effect in milliseconds.

    Returns:
        np.ndarray: The audio data with the fade-in effect applied. It's a copy of the original audio data, which is not modified.
    """
    fade_samples = int(sample_rate * fade_duration_ms / 1000)
    fade = np.linspace(min_level, max_level, fade_samples, dtype=np.float32)
    # Create a copy of the audio data

    if not in_place:
        audio_data = audio_data.copy()

    # Apply fade to the beginning of the loop
    audio_data[:, :fade_samples] *= fade

    return audio_data

def fade_out(audio_data: np.ndarray, sample_rate: int, fade_duration_ms: int, min_level: float = 0.0, max_level: float = 1.0, in_place: bool = False) -> np.ndarray:
    """ Applies a fade-out effect to the start of an audio segment.

    Args:
        audio_data (np.ndarray): The audio data to fade out.
        sample_rate (int): The sample rate of the audio data.
        fade_duration_ms (int): The duration of the fade-out effect in milliseconds.

    Returns:
        np.ndarray: The audio data with the fade-out effect applied. It's a copy of the original audio data, which is not modified.
    """
    fade_samples = int(sample_rate * fade_duration_ms / 1000)
    fade = np.linspace(max_level, min_level, fade_samples, dtype=np.float32)

    if not in_place:
        audio_data = audio_data.copy()

    # Apply fade to the end of the loop
    audio_data[:, -fade_samples:] *= fade

    return audio_data


def nearest_zero_crossing(audio_data: ndarray, start_index: int) -> int:
    """ Find the nearest zero-crossing around a given index. 

        Args:
            audio_data (ndarray): The audio data to search.
            start_index (int): The index to start searching from.

        Returns:
            int: The index of the nearest zero-crossing, or -1 if none is found.
    """
    if start_index >= len(audio_data):
        return -1
    start = start_index if start_index > 0 else 1
    index = -1
    dist = -1
    # prefer zero-crossings to the right of the start index
    for i in range(start, len(audio_data)):
        if audio_data[i] * audio_data[i-1] <= 0:
            index = i if abs(audio_data[i]) < abs(audio_data[i-1]) else i-1
            dist = i - start_index
            break

    if start > 1:
        for i in range(start-1, 0, -1):
            # stop searching if the distance to the start index is greater than the distance to the previously found zero-crossing
            if dist >= 0 and start_index - i >= dist:
                break
            if audio_data[i] * audio_data[i-1] <= 0:
                index = i if abs(audio_data[i]) < abs(audio_data[i-1]) else i-1
                break
    return index


def spectral_similarity(audio_data: ndarray, sample_rate: int, start: int, end: int) -> float:
    """Check the spectral similarity between loop start and end points.

        Args:
            audio_data (ndarray): The audio data to search (should be single channel, i.e. mono!).
            sample_rate (int): The sample rate of the audio data.
            start (int): The index to start searching from.
            end (int): The index to end searching at.
        Returns:
            float: The Pearson correlation coefficient between the start and end points.
    """
    # Adjust frame_length and hop_length based on sample rate based on 44.1kHz as standard
    frame_length = int(2048 * (sample_rate / 44100))
    hop_length = int(512 * (sample_rate / 44100))

    start_data = audio_data[start:start + frame_length]
    if len(start_data) < frame_length:
        raise ValueError(
            "The segment is too short for the given frame length.")
    start_spec = librosa.feature.melspectrogram(
        y=start_data, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)
    end_data = audio_data[end - frame_length:end]
    if len(end_data) < frame_length:
        raise ValueError(
            "The segment is too short for the given frame length.")
    end_spec = librosa.feature.melspectrogram(
        y=end_data, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)

    # Pearson correlation coefficient between start and end, i.e. element_0,1 in the 2x2 matrix
    similarity = np.corrcoef(start_spec.flat, end_spec.flat)[0, 1]
    return similarity


def find_similar_endpoints(audio_data: ndarray, sample_rate: int, frames: ndarray, threshold: float = 0.8, max_frames: int = 120) -> Tuple[int, int]:
    """ Find similar endpoints from both sides of an audio segment (mono!).

    Args:
        audio_data (ndarray): The audio data to search
        sample_rate (int): The sample rate of the audio data.
        frames (ndarray): The frames to search.
        threshold (float, optional): The Similarity threshold to use when comparing. Defaults to 0.8.
        max_frames (int, optional): Maximum number of frames to search around a point. Defaults to 120.

    Returns:
        Tuple[int, int]: The first 2 endpoints found, or (-1, -1) if none were found.
    """
    for i in range(min(len(frames) - 1, max_frames)):
        start = librosa.frames_to_samples(frames[i])
        start = nearest_zero_crossing(audio_data=audio_data, start_index=start)
        if start >= 0:
            for j in range(min(len(frames) - 1 - i, max_frames)):
                end_index = len(frames) - 1 - j
                end = librosa.frames_to_samples(frames[end_index])
                end = nearest_zero_crossing(
                    audio_data=audio_data, start_index=end)
                if end >= 0:
                    # check similarity
                    try:
                        similarity = spectral_similarity(
                            audio_data=audio_data, sample_rate=sample_rate, start=start, end=end)
                        if similarity > threshold:
                            if start >= 0 and end >= 0:
                                return start, end
                    except ValueError:
                        break  # start - end segment is too short
                else:
                    break
        else:
            break
    return -1, -1


def prune_silence(audio: AudioData, min_silence_ms: int = 1000, keep_silence_ms: int = 100, top_db: float = 60) -> AudioData:
    """ Cleans up the silent parts of an audio segment.

        The logic is like this:
        1. Trims all the silence from the start and end of the audio 
        2. Replace any internal silence intervals longer than `min_silence_ms` with silence of `keep_silence_ms` milliseconds.

        The threshold for silence is determined by `top_db` which is the threshold (in decibels) below reference to consider as silence 
        where `reference` is the maximum across the whole audio.

    Args:
        audio_data (np.ndarray): The audio data to trim, expected in float32 format with values ranging from -1.0 to 1.0.
        sample_rate (int): The sample rate of the audio data.
        min_silence_ms (int): The minimum length of silence to detect (in milliseconds).
        keep_silence_ms (int): The amount of silence to keep before the non-silent chunks audio (in milliseconds).
        top_db (float): The threshold (in decibels) below reference to consider as silence.

    Returns:
        np.ndarray: Trimmed audio data.
    """
    if keep_silence_ms < 0 or min_silence_ms < 0 or keep_silence_ms >= min_silence_ms:
        raise ValueError("Invalid silence duration parameters.")

    logger = logging.getLogger("global")
    logger.debug("Pruning silence from audio. Initial duration: %dms", audio.duration)
    
    # Trim silence from the start and end of the audio
    audio_data, _ = librosa.effects.trim(audio.audio_data, top_db=top_db)
    logger.debug("Trimmed silence from the ends. New duration: %dms", (len(audio_data) if audio_data.ndim == 1 else audio_data.shape[1]) * 1000 // audio.sample_rate)
    
    # Convert silence duration from milliseconds to number of samples
    min_silence = int(audio.sample_rate * min_silence_ms / 1000)
    keep_silence = int(audio.sample_rate * keep_silence_ms / 1000)

    # Silence analysis will be performed on the mono audio data
    # while the trimming will be performed on the original audio data (stereo or mono)
    channels = 1
    if audio_data.ndim == 2 and audio_data.shape[0] == 2:
        audio_data_mono = np.mean(audio_data, axis=0)
        channels = 2
    else:
        audio_data_mono = audio_data

    # Detect non-silent intervals
    non_silent_intervals = librosa.effects.split(
        audio_data_mono, top_db=top_db)
    
    # filterout intervals shorter than min_silence
    filtered_intervals = []
    for start, end in non_silent_intervals:
        logger.debug("Non-silent interval: %ds - %ds", start/audio.sample_rate, end/audio.sample_rate)
        # Update prev_end to the end of the last interval in filtered_intervals
        prev_end = filtered_intervals[-1][1] if filtered_intervals else 0
        if start > 0:
            if start - prev_end > min_silence:
                logger.debug("Keeping interval: %ds - %ds", (start - keep_silence) / audio.sample_rate, end / audio.sample_rate)
                # Add interval with silence wrapped around
                filtered_intervals.append((start - keep_silence, end))
            else:
                # Update the end of the last interval
                logger.debug("Expanding previous non-silent interval end from %ds to %ds", filtered_intervals[-1][1] / audio.sample_rate, end / audio.sample_rate)
                filtered_intervals[-1] = (filtered_intervals[-1][0], end)
        else:
            filtered_intervals.append((0, end))

    # Initialize a list to hold processed audio for each channel
    processed_audio_channels = []

    # Process each channel based on the filtered intervals
    processed_audio_1 = []
    processed_audio_2 = []
    for start, end in filtered_intervals:
        logger.debug("Keeping interval %ds - %ds", start / audio.sample_rate, end / audio.sample_rate)
        # Append the non-silent audio chunk for this channel
        processed_audio_1.append(audio_data[0, start:end])
        if channels > 1:
            processed_audio_2.append(audio_data[1, start:end])
    # Concatenate processed chunks and add to the channel list
    processed_audio_channels.append(np.concatenate(processed_audio_1, axis=0))
    if channels > 1:
        processed_audio_channels.append(
            np.concatenate(processed_audio_2, axis=0))

    # Stack processed channels back into multi-channel format
    return AudioData(np.stack(processed_audio_channels, axis=0), audio.sample_rate)

def adjust_loop_ends(loop: AudioData, loop_start: int, loop_end: int) -> tuple[int, int]:
    """Adjust the loop using the proximity of onsets to the loop ends
    
    Args:
        audio (AudioData): The audio data to analyze
        loop_start (int): The start index of the loop
        loop_end (int): The end index of the loop
        
    Returns:
        tuple[int, int]: The adjusted loop start and end indices
    """
    mono_samples = loop.mono_audio_data[0]
    # Analyze onset strength
    onset_env = librosa.onset.onset_strength(y=mono_samples, sr=loop.sample_rate)
    # Find onsets
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=loop.sample_rate, units='samples')
    
    if len(onsets) > 0:
        # Find the closest onsets
        start_onset = min(onsets, key=lambda x:abs(x-loop_start))
        end_onset = min(onsets, key=lambda x:abs(x-loop_end))
        if start_onset >= 0:
            loop_start = start_onset
        if end_onset < loop.length:
            loop_end = end_onset
    return loop_start, loop_end

def align_phase(audio_data: ndarray, loop_start: int, loop_end: int, segment_length: int=2048, in_place = False, crossfade_min: float = 0, crossfade_max: float = 1.0) -> ndarray:
    """
    Aligns phase between the start and end of a loop within a multi-channel audio signal.
    
    Parameters:
    audio_data (np.ndarray): The multi-channel audio data array of shape (C, S).
    loop_start (int): The starting sample index of the loop.
    loop_end (int): The ending sample index of the loop.
    segment_length (int): Length of the segments to be used for phase alignment.
    
    Returns:
    np.ndarray: The phase-aligned audio data.
    """
    # Validate loop indices
    if loop_start < 0 or loop_end >= audio_data.shape[1] or loop_end <= loop_start:
        raise ValueError("Invalid loop_start or loop_end indices")
    
    if not in_place:
        audio_data = audio_data.copy()
    
    sample_count = audio_data.shape[1]
    # Ensure the segment length does not exceed the loop length and available samples after loop_end
    segment_length = min(segment_length, sample_count)
    
    segment_a_start = min(max(loop_start - segment_length//2, 0), sample_count - segment_length)
    segment_b_start = min(max(loop_end - segment_length//2, 0), sample_count - segment_length)

    # Iterate over each channel
    for channel in range(audio_data.shape[0]):
        # Extract segments for phase comparison
        segment_a = audio_data[channel, segment_a_start:segment_a_start + segment_length]
        segment_b = audio_data[channel, segment_b_start:segment_b_start + segment_length]
        
        # Compute the FFT of both segments
        fft_a = np.fft.fft(segment_a)
        fft_b = np.fft.fft(segment_b)
        
        # Calculate the phase difference and adjust the end segment
        phase_diff = np.angle(fft_a) - np.angle(fft_b)
        adjusted_fft_b = np.abs(fft_b) * np.exp(1j * (np.angle(fft_b) + phase_diff))
        
        # Apply inverse FFT to get the time-domain signal back
        adjusted_segment_b = np.real(np.fft.ifft(adjusted_fft_b))
        
        # Crossfade blending
        half_pi = np.pi / 2
        t = np.linspace(crossfade_min * half_pi, crossfade_max * half_pi, segment_length)
        fade_out = np.cos(t)  # Decreases in volume
        fade_in = np.sin(t)   # Increases in volume

        original_segment = audio_data[channel, segment_b_start:segment_b_start + segment_length]
        
        blended_segment = original_segment * fade_out + adjusted_segment_b * fade_in
        
        audio_data[channel, segment_b_start:segment_b_start + segment_length] = blended_segment

    return audio_data

def export_audio(audio: AudioData, filename_base: str, format="wav"):
    """ Write the audio data to a WAV file.

        Args:
            audio_data (ndarray): The audio data to write.
            sample_rate (int): The sample rate of the audio data.
            filename (str): The name of the file to write to (without extension, it will automatically add .wav).
    """
    wav = torch.from_numpy(audio.audio_data)
    # audo_write will create the directory chain if it doesn't exist
    audio_write(filename_base, wav, audio.sample_rate,
                strategy="loudness", loudness_compressor=True, format=format, make_parent_dir=True)

def set_all_seeds(seed):
    # From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def calculate_checksum(data: bytes):
    return hashlib.md5(data).hexdigest()

def setup_logging(logs_file:str = "global.log", logs_path: str = os.path.join(".", "logs"), log_level: int = logging.INFO) -> logging.Logger:
    log_file = os.path.join(logs_path, logs_file)

    # Create log directory if it doesn't exist
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)

    # Create a logger
    logger = logging.getLogger('global')
    logger.setLevel(logging.INFO if log_level is None or log_level < 0 else log_level)

    # Create a handler that writes log messages to a file, with log rotation
    handler = RotatingFileHandler(
        log_file, maxBytes=1024*1024*5, backupCount=5)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger