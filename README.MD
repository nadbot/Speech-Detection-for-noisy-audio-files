# Voice Activity Detection in noisy environments

---

## Structure

This project contains Jupyter notebooks as well as python files.
The python files are used for methods used across multiple notebooks.
The Jupyter notebooks are the main component and actually run the Voice Activity Detection.

### Jupyter Notebooks

There are currently four Jupyter notebooks:
 - CNN_VAD
   - (use the CNN-VAD model from [1] for VAD)
 - U-net
   - (use the U-net model from [2] for VAD)
 - Wav2vec
   - (transform the Wav2vec model for ASR [3] for VAD)
   - created as main part of this project
 - Speech_Emotion_Recognition
   - (transform the Wav2vec model for SER)

See the corresponding notebook for more information about the model used.


### Python Files

These python files are currently used:
 - configs_private
   - Private configuration for connecting to databricks
   - Used for storing the model results using mlflow
 - Enums
   - Multiple Enums used for getting the data and setting the feature used
 - Parameters
   - Dataclass for setting all parameters used for feature extraction (e.g. window size, nr of MFCC)
   - Currently has U-net style features, as well as Spectrograms for different window sizes
 - get_files
   - Get all files from local QUT-NOISE-TIMIT directory for the given SNR and recording session
 - create_features_and_labels
   - Create the features and corresponding labels for a given audio-filename and parameters
 - plot_files
   - Plot the distribution of the audio files per location and label
 - process_frames_wav2vec
   - Further processing step for Wav2vec to include input_values etc as dictionary
 - evaluation_tensorflow
   - Method to do all wanted evaluation and show plots
   - Only works for the tensorflow models so far (CNN-VAD and U-net)


## Example workflow

### Get Files:
To get the list of audiofiles for a certain noise level, simply run the get_available_files function
```python
from Enums import AudioClipFileLength, RecPlace, SNR
from get_files import get_available_files
snr = SNR.MINUS_FIVE  # SNR value of the current experiment
wav_files_train = get_available_files(path_files+"QUT-NOISE-TIMIT-new/", snr, AudioClipFileLength.ONE_MINUTE, rec_place=RecPlace.a)
```

### Set Parameters
```python
from Enums import FeatureType
from Parameters import Parameters
# set all parameters as wanted
parameter = Parameters()
# set_params_specs is the default for Spectrogram
if feature == FeatureType.SPECTROGRAM or feature == FeatureType.RAW_AUDIO:
    print('Using spectrogram or raw audio parameter')
    parameter.set_params_specs()
else:
    print('Using MFCC parameter')
    parameter.set_params_unet()
```

### Extract Features
```python
from create_features_and_labels import get_feature_and_labels
from tensorflow.keras.utils import to_categorical

ignore_raw_audio = True  # if raw audio is needed
end_train = 100  # how many files should be used
x_train, labels_train, frames_pos_train, frames_train = get_feature_and_labels(wav_files_train[0:end_train], parameter, ignore_raw_audio=ignore_raw_audio)
y_train = to_categorical(labels_train)
```
If end_train is not equal to the length of the dataset, plot_files.plot_file_distribution can be used to
show the distribution across categories.

### Process Frames
If the Wav2vec model is used, the model expects a dataset, where each input is a dictionary
containing fields for the labels and for the input_values as well as additional inputs.
Therefore, run the function below to convert the numpy arrays to this structure.

```python
from process_frames_wav2vec import get_processed_frames_VAD

train_dataset = get_processed_frames_VAD(frames_train, x_train, y_train)
```

### Run model
Compile and run the wanted model as done in the jupyter notebooks.

### Evaluation
Evaluate the results of the model on the test set.
If a path of the best weights is given, these will be automatically used, otherwise the last weights are used.
This function will print the training (and eval) accuracy and loss using the history,
the confusion matrices with and without post-processing and the resulting HTER.
```python
from evaluation_tensorflow import Evaluation
eval = Evaluation(model, run, history, x_test, y_test, snr, parameter,
                  model_name, best_weights, save_figures=True)
eval.evaluate()
```

### Sources
[1] Diego Augusto Silva et al. “Exploring convolutional neural networks for voice activity detection”.
In: Cognitive Technologies. Springer, 2017, pp. 37–47.

[2] Aleksei Gusev et al. “Deep speaker embeddings for far-field speaker recognition on short utterances”.
In: arXiv preprint arXiv:2002.06033 (2020)

[3] Alexei Baevski et al. “wav2vec 2.0: A framework for self-supervised learning of speech representations”.
In: arXiv preprint arXiv:2006.11477 (2020).
