from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from librosa.feature import melspectrogram, mfcc
from scipy.signal import get_window
from scipy.signal import wiener
from tqdm import tqdm

from Enums import FeatureType
from Parameters import Parameters


def create_labels(window_start: int, labels: np.ndarray, parameters: Parameters, emotion: bool = False) -> \
        Tuple[List[int], List[Dict[str, int]]]:
    """
    Function to create labels from the sliding window starting at window_start

    Args:
        window_start: Start index of the window, end is calculated from parameters
        labels: The ndarray of labels for the whole audio file
        parameters: The Parameters for the given run, used to calculate Window Size and overlap
        emotion: Whether the problem is emotion (multi-label) or VAD (single-label)

    Returns:
        List of labels per frame of the window
        List of positions of the label in the audio file
    """
    labels_frame = []
    # As the audio is padded, pad the labels as well.
    padding: int = int(parameters.n_fft/2)
    start: int = max(window_start-padding, 0)
    end: int = min(parameters.sliding_window_size + padding, len(labels))
    frame_positions_window: List[Dict[str, int]] = []
    if emotion:
        # this emotion is extracted per window, not per frame
        label_window = labels[start:end]
        unique, counts = np.unique(label_window[np.nonzero(label_window)], return_counts=True)
        if len(counts) > 0:
            most_found_label = unique[counts.argmax()]
        else:
            most_found_label = 0
        return most_found_label, [{'lower': start, 'upper': end}]
    for frame_nr in range(start, start+end, parameters.hop_length):
        # The size of each label range is 1 FFT size
        lower_border = min(max(frame_nr-int(parameters.n_fft/2), 0), len(labels))
        upper_border = min(frame_nr+int(parameters.n_fft/2), len(labels))
        label_frame = labels[lower_border:upper_border]
        frame_positions_window.append({'lower': lower_border,
                                       'upper': upper_border})
        # declare as speech if at least some samples are speech
        sums = np.sum(label_frame, axis=0)
        sums = sums >= 1
        labels_new = sums.astype(int)
        labels_frame.append(labels_new)
    return labels_frame[:-1], frame_positions_window[:-1]


def create_labels_from_eventlab(filename: str, parameters: Parameters,
                                print_df: bool = False) -> np.ndarray:
    """
    Create the labels by reading the eventlab file
    and converting the samples where speech is present to 1 and
    without speech to 0.
    Example eventlab file:
    Start   End   Label
    0.5     3.21  speech
    3.21    45.12 no speech
    45.12   60    speech
    """
    eventlab = filename.replace(".wav", ".eventlab")
    df = pd.read_csv(eventlab, sep=" ", header=None)
    df = df.loc[df[2] == 'speech'].copy()
    # convert start and end time of speech to samples instead of seconds
    if print_df:
        print(df)
    df[0] *= parameters.sr
    df[1] *= parameters.sr
    labels = np.array([0] * 960001)
    for index, row in df.iterrows():
        labels[round(row[0]):round(row[1])+1] = 1
    return labels


def create_emotion_labels(filename: str, parameters: Parameters) -> np.ndarray:
    """
    Create the labels by reading the eventlab and timit file
    and converting the samples where speech is present to the correct emotion.
    Example eventlab file:
    Start   End   Label
    0.5     3.21  speech
    3.21    45.12 no speech
    45.12   60    speech
    Example timitlab file:
    0 3.73704 Actor_23/03-01-01-01-01-02-23.wav
    25.1148 28.5516 Actor_04/03-01-04-01-02-01-04.wav
    28.1875 32.0247 Actor_10/03-01-03-02-01-01-10.wav
    """
    eventlab: str = filename.replace(".wav", ".eventlab")
    timitlab: str = filename.replace(".wav", ".timitlab")
    label = np.array([0] * 960001)
    df = pd.read_csv(eventlab, sep=" ", header=None)
    df = df.loc[df[2] == 'speech'].copy()
    if df.shape[0] > 0:
        # this line fails if no speech exists, as it tries to load an empty file.
        # So instead use the other file to see if speech exists
        df_timit = pd.read_csv(timitlab, sep=" ", header=None)

        # df_timit looks like this, so no filtering for speech needed:
        # 18.7486 22.2855 Actor_14/03-01-04-01-02-02-14.wav
        # 40.0766 43.9805 Actor_14/03-01-05-01-02-02-14.wav
        # 48.8286 52.7325 Actor_20/03-01-06-02-02-02-20.wav
        df_timit[0] *= parameters.sr
        df_timit[1] *= parameters.sr
        # label = np.array([0] * audio_file_length)
        for index, row in df_timit.iterrows():
            file_name = row[2].split('/')[1].split('.')[0]
            # Filename identifiers
            # Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
            # Vocal channel (01 = speech, 02 = song).
            # Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
            # Emotional intensity (01 = normal, 02 = strong). NOTE:
            #   There is no strong intensity for the 'neutral' emotion.
            # Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
            # Repetition (01 = 1st repetition, 02 = 2nd repetition).
            # Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
            ravdess_identifiers = file_name.split('-')
            # for now we are only interested in the emotion itself, so just add this one
            emotion = int(ravdess_identifiers[2])
            if emotion == 1:
                # merge neutral and calm emotion as done by Pepino et al.
                emotion = 2
            emotion = emotion - 1
            # new mapping:
            # Emotion: 00=no speech, 01=(calm/neutral), 02=happy, 03=sad, 04=angry, 05=fearful, 06=disgust, 07=surprised
            label[round(row[0]):round(row[1])+1] = emotion
    return label


def create_feature(audio_segment: np.ndarray, parameters: Parameters) -> np.ndarray:
    """
    Create feature from the audio_segment using the parameters given.
    Can either be MFCC or Spectrogram depending on the parameters used.
    """
    if parameters.feature_type == FeatureType.MFCC:
        feature = mfcc(y=audio_segment,
                       sr=parameters.sr,
                       n_fft=parameters.n_fft,
                       n_mfcc=parameters.n_mfcc,
                       n_mels=parameters.n_mels,
                       hop_length=parameters.hop_length,
                       fmin=parameters.fmin,
                       fmax=parameters.fmax,
                       htk=parameters.htk,
                       center=parameters.center)[:, :-1]
    elif parameters.feature_type == FeatureType.SPECTROGRAM:
        feature = melspectrogram(audio_segment,
                                 sr=parameters.sr,
                                 n_fft=parameters.n_fft,
                                 n_mels=parameters.n_mels,
                                 hop_length=parameters.hop_length,
                                 center=parameters.center,
                                 fmax=parameters.fmax)[:, :-1]
    else:
        raise Exception('Invalid feature type given')
    return feature


def padded_audio(audio: np.ndarray, parameters: Parameters, overlap: float = 0.5,
                 remove: bool = False) -> np.ndarray:
    """
    Function to pad the audio to be a multiple of the window size.
    Can either remove the last few samples or create empty ones
    """

    padding_needed = len(audio) % (parameters.sliding_window_size * overlap)

    if remove:
        new_length = len(audio) - padding_needed
        new_audio = [0] * int(new_length)
        new_audio[0:new_length] = audio[0:new_length]
    else:
        new_length = (parameters.sliding_window_size * overlap) \
                     - padding_needed + len(audio)
        new_audio = [0] * int(new_length)
        new_audio[0: len(audio)] = audio

    new_audio = np.array(new_audio)

    return new_audio


def get_feature_and_labels(wav_files: List[str], parameters: Parameters, ignore_raw_audio: bool = False,
                           emotion: bool = False) -> Tuple[np.ndarray, np.ndarray,
                                                           List[List[Dict[str, int]]], np.ndarray]:
    """
    Main function to get the features and labels from a given list of audio filepaths.
    """
    features_extracted = []
    labels_extracted = []
    index = 0
    frames_all = []
    frame_positions = []
    for filename in tqdm(wav_files):
        frame_positions = []  # should be equal for every file, so just overwrite previous one

        index += 1
        audio, sr = librosa.load(filename, sr=parameters.sr)

        audio: np.ndarray = padded_audio(audio, parameters)

        if parameters.use_wiener_filter and index == 1:
            print(f"Using {parameters.wiener_filters} Wiener Filter")
            audio = wiener(audio, parameters.wiener_filters)

        # labels
        if emotion:
            labels: np.ndarray = create_emotion_labels(filename, parameters)
        else:
            labels: np.ndarray = create_labels_from_eventlab(filename, parameters)

        window_start = 0
        overlap = int(parameters.sliding_window_size/2)
        while window_start + overlap < len(audio)-1:
            # get the window
            audio_window: np.ndarray = audio[window_start: window_start +
                                             parameters.sliding_window_size]
            hann_window = get_window("hann", parameters.sliding_window_size,
                                     fftbins=True)
            audio_window = audio_window * hann_window
            feature = create_feature(audio_window, parameters)
            labels_frame, frame_positions_window = create_labels(window_start,
                                                                 labels,
                                                                 parameters,
                                                                 emotion)
            if not ignore_raw_audio:
                # reduce computation by not adding the raw samples unless necessary
                frames_all.append(audio_window)
            features_extracted.append(feature)
            labels_extracted.append(labels_frame)
            window_start += overlap
            frame_positions.append(frame_positions_window)
    frames_all = np.array(frames_all)
    features_extracted = np.array(features_extracted)
    labels_extracted = np.array(labels_extracted)
    return features_extracted, labels_extracted, frame_positions, frames_all
