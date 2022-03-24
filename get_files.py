import glob
from typing import List

from Enums import AudioClipFileLength, RecPlace, SNR


def get_available_files(path_base: str, snr: SNR = SNR.ZERO,
                        audio_clip_length_files: AudioClipFileLength = AudioClipFileLength.ONE_MINUTE,
                        rec_place: RecPlace = RecPlace.a) -> List[str]:
    """
    Function that gets all filenames for a certain SNR, rec_place and audio_clip_length_files.

    Args:
        path_base: Path to the QUT-NOISE-TIMIT data
        snr: SNR that shall be used. Getting files across multiple snrs is not supported
        audio_clip_length_files: Length of the audio clip, usually 1min is used.
        rec_place: Place where the recording was taken, either A or B.

    Returns:
        List of strings with the audio file names
    """
    # get the paths of all locations (street, car, etc)
    locations: List[str] = glob.glob(path_base + "*/")
    wav_files: List[str] = []

    for folder in locations:
        # create the path using the arguments given
        path = folder + rec_place.value + str(audio_clip_length_files.value) + snr.value
        # find all wav files inside that path
        wavs = glob.glob(path + "/*.wav")
        if len(wavs) > 1:
            # if the folder is empty continue, otherwise save all filenames
            wav_files.extend(wavs)

    print(f'Total length: {len(wav_files)}')
    return wav_files
