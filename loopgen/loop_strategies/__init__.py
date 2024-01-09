""" A module with implementations for different loop generation strategies. 
"""
from .base import LoopStrategy
from .beat_aligned import BeatAligned
from .transient_aligned import TransientAligned
from .crossfade import CrossFade
from .fade_in_out import FadeInOut