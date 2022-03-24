from dataclasses import dataclass

from Enums import FeatureType


@dataclass
class Parameters:
    """Dataclass to easily access and change all parameters for the
    MFCC extraction."""

    window_size: int = 0
    sliding_window_size: int = 0
    hop_length: int = 0
    n_mfcc: int = 0
    n_mels: int = 0
    n_fft: int = 0
    fmin: int = 0
    fmax: int = None
    sr: int = 16000
    htk: bool = False
    center: bool = True
    wiener_filters: int = 0
    use_wiener_filter: bool = False
    feature_type: FeatureType = FeatureType.MFCC

    def __init__(self):
        self.set_params_unet()

    def set_params_unet(self):
        self._set_default()
        self.n_mfcc = 32  # number of MFCCs to return (changed to 32 instead of 23 for easier Unet model)
        self.n_mels = 40  # number of Mel bands to generate
        self.feature_type = FeatureType.MFCC

    def set_params_specs(self):
        self._set_default()

    def set_params_640_specs(self):
        self._set_default()
        n_fft = 160  # 160 # 80 # 400=25ms
        self.window_size: int = n_fft
        self.n_fft = n_fft  # number of FFT components
        self.hop_length = 80  # 80=5ms
        self.sliding_window_size = int(16000/1000*640)

    def set_params_320_specs(self):
        self._set_default()
        n_fft = 80  # 80=5ms
        self.window_size: int = n_fft
        self.n_fft = self.window_size  # number of FFT components
        self.hop_length = 40  # 40=2.5ms
        self.sliding_window_size = int(16000/1000*320)

    def set_params_160_specs(self):
        self._set_default()
        n_fft = 40  # 40=2.5ms
        self.window_size: int = n_fft
        self.n_fft = self.window_size  # number of FFT components
        self.hop_length = 20  # 20=1.25ms
        self.sliding_window_size = int(16000/1000*160)

    def _set_default(self):
        """
        Default choice if the spectrogram is chosen.
        This function uses 128 labels for each 2.56s and has as features 128 spectrograms with 32 mels each.
        Each frame of the 2.56s window is 25ms and a hop length of 20ms is used.

        These parameters are similar to the U-Net implementation used by Gusev et al.
        The main differences are:
            - the number of mels is 32 instead of 23
            - the feature is spectrogram instead of U-Net
        """
        # As explained in their paper:
        #   window_size = 25ms with 5ms overlap
        #   The model takes 8kHz 23-dimensional MFCC features as input
        #   Our VAD solution works with a half overlapping 2.56sec sliding window and a 1.28sec overlap.
        #   It should be noted that each MFCC vector is extracted for 25ms frame every 20ms.
        #   This results in 128 × 23 input features size for the neural network (2.56s*1000/20ms)

        # all values below are in samples not ms
        self.window_size: int = 400
        self.sliding_window_size = int(2.56 * 16000)
        self.hop_length = int(20 * 16000 / 1000)  # number of samples between successive frames. See librosa.stft
        self.n_fft = self.window_size  # number of FFT components

        # self.n_mfcc = 32  # number of MFCCs to return (changed to 32 instead of 23 for easier Unet model)
        self.n_mels = 32  # number of Mel bands to generate
        self.fmin = 0  # lowest frequency (in Hz)
        self.fmax = 8000  # highest frequency (in Hz). If None, use fmax = sr / 2.0
        self.sr = 16000
        self.htk = False  # use HTK formula instead of Slaney
        self.center = True  # if true, pads the first and last element, thus creating 129 elements

        self.wiener_filters = 12
        self.use_wiener_filter = False
        self.feature_type = FeatureType.SPECTROGRAM
