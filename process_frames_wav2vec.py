import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor


def get_processed_frames_VAD(frames, rgb_specs, labels, emotion: bool = False):
    """
    Wav2vec specific for VAD

    """
    if emotion:
        return get_processed_frames_emotion(frames, rgb_specs, labels)
    # Use the standard wav2vec model as base for setting the parameter
    model_base = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_base)
    target_sampling_rate = processor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")
    complete_list = []
    result = processor(frames, sampling_rate=16000, do_normalize=True, return_attention_mask=True)
    for frame in tqdm(range(frames.shape[0])):
        frame_dict = {}
        label = np.argmax(labels[frame], axis=1)
        frame_dict["labels"] = []
        for l in label:
            if l == 0:
                frame_dict["labels"].append('no speech')  # labels for that frame
            else:
                frame_dict["labels"].append('speech')  # labels for that frame
        # get counts for each label to get the majority label of that frame
        unique, counts = np.unique(label, return_counts=True)
        if np.argmax(counts) == 0:
            frame_dict["majority_label"] = 'no speech'
        else:
            frame_dict["majority_label"] = 'speech'
        frame_dict['input_values'] = result['input_values'][0][frame]  # input values for that frame
        frame_dict['raw_audio'] = frames[frame]
        frame_dict['rgb_spectrogram'] = rgb_specs[frame].astype(np.float32)
        frame_dict['attention_mask'] = result['attention_mask'][0][frame]
        complete_list.append(frame_dict)
    return complete_list


def get_processed_frames_emotion(frames, rgb_specs, labels):
    """
    Wav2vec specific for SER
    """
    model_base = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_base)
    target_sampling_rate = processor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")
    # frames = frames.reshape(frames.shape[0]*frames.shape[1], frames.shape[2])
    complete_list = []
    result = processor(frames, sampling_rate=16000, do_normalize=True, return_attention_mask=True)
    for frame in tqdm(range(frames.shape[0])):
        frame_dict = {}
        # label = np.argmax(labels[frame], axis=1)
        label = labels[frame]
        frame_dict["labels"] = []
        # # Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        # emotion_labels = ['no speech', 'neutral', 'calm', 'happy',
        #                   'sad', 'angry', 'fearful', 'disgust', 'surprised']
        frame_dict['labels'] = label
        frame_dict['majority_label'] = label
        frame_dict['input_values'] = result['input_values'][0][frame]  # input values for that frame
        frame_dict['rgb_spectrogram'] = rgb_specs[frame].astype(np.float32)
        frame_dict['attention_mask'] = result['attention_mask'][0][frame]
        complete_list.append(frame_dict)
    return complete_list
