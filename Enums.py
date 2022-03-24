from enum import Enum


class AudioClipFileLength(Enum):
    """ QUT-Noise-TIMIT produces two lengths for audio,
    60 seconds and 120 seconds. For now, only 60s will be used."""
    ONE_MINUTE = '/l060/'
    TWO_MINUTES = '/l120/'


class RecPlace(Enum):
    """ Recording place, one is used for training, the other for testing. """
    a = 'sA'
    b = 'sB'


class SNR(Enum):
    """ Noise level of audio files """
    MINUS_TWENTY = 'n-20'
    MINUS_FIFTEEN = 'n-15'
    MINUS_TEN = 'n-10'
    MINUS_FIVE = 'n-05'
    ZERO = 'n+00'
    FIVE = 'n+05'
    TEN = 'n+10'
    FIFTEEN = 'n+15'


class FeatureType(Enum):
    """ Type of feature that shall be used, raw audio is usually used in combination with one of the others."""
    MFCC = 'MFCC'
    SPECTROGRAM = 'Spectrogram'
    RAW_AUDIO = 'Raw Audio'
