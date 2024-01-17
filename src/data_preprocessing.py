import os
import pathlib
from typing import Optional

import mne
import numpy as np
import scipy.interpolate
import scipy.io
import tqdm


class PreprocessingPipeline:
    """
    Preprocessing pipeline for the ECoG data.

    :param save_dir: Directory to save the preprocessed data
    :param time_delay_secs: Time delay hyperparameter
    :param raw_data_dir: Directory to load the raw data
    :param kwargs: Additional parameters to pass to the pipeline

    :ivar ecog_train_data: Preprocessed ECoG training data
    :ivar fingerflex_train_data: Preprocessed finger motions training data
    :ivar ecog_test_data: Preprocessed ECoG test data
    :ivar fingerflex_test_data: Preprocessed finger motions test data
    :ivar save_dir: Directory to save the preprocessed data
    :ivar raw_data_dir: Directory to load the raw data
    :ivar kwargs: Additional parameters to pass to the pipeline


    The pipeline consists of the following steps:
    0. Downloading the raw data if it is not already downloaded
    1. Interpolation of finger motions to match the sampling rate of the ECoG data
    2. Standardization and removal of the median from each channel
    3. Harmonics removal and frequency filtering
    4. Computation of spectrogramms using wavelet transforms
    5. Reducing the sampling rate of spectrograms
    6. Optional conversion to db, not used in the final version
    7. Taking into account the delay between brain waves and movements
    8. Scaling of finger motions and ECoG data
    9. Saving the preprocessed data

    The pipeline can be run using the run() method, which takes the following parameters:
    :param save_dir: Directory to save the preprocessed data
    :param time_delay_secs: Time delay hyperparameter
    :param mat_path: Path to the .mat file with the training data
    :param data_key: Key to the training data in the .mat file
    :param dg_key: Key to the finger motions in the .mat file
    :param reshape: Whether to reshape the data to (features, time)
    :param kwargs: Additional parameters to pass to the pipeline

    The pipeline can be loaded using the load() method, which takes the following parameters:
    :param load_dir: Directory to load the preprocessed data
    :param kwargs: Additional parameters to pass to the pipeline

    The pipeline can be saved using the save() method, which takes the following parameters:
    :param save_dir: Directory to save the preprocessed data
    :param reshape: Whether to reshape the data to (features, time)
    :param kwargs: Additional parameters to pass to the pipeline

    For now the data can be downloaded from
    https://stacks.stanford.edu/file/druid:zk881ps0522/BCI_Competion4_dataset4_data_fingerflexions.zip
    and placed in the :param raw_data_dir: directory. If the data is not downloaded, the pipeline will download
    it automatically with the download_raw_data() or at the first run() call.
    """
    L_FREQ, H_FREQ = 40, 300  # Lower and upper filtration bounds
    CHANNELS_NUM = 62  # Number of channels in ECoG data
    WAVELET_NUM = 40  # Number of wavelets in the indicated frequency range, with which the convolution is performed
    DOWNSAMPLE_FS = 100  # Desired sampling rate
    DEFAULT_TIME_DELAY_SECS = 0.2  # Default Time delay hyperparameter

    TR_DATA_KEY = 'train_data'
    TR_DG_KEY = 'train_dg'
    TR_DEFAULT_MAT = 'sub1_comp.mat'
    TEST_DATA_KEY = 'test_data'
    TEST_DG_KEY = 'test_dg'
    TEST_DEFAULT_MAT = 'sub1_testlabels.mat'
    RAW_DATA_URL = r"https://stacks.stanford.edu/file/druid:zk881ps0522/BCI_Competion4_dataset4_data_fingerflexions.zip"
    DEFAULT_RAW_DATA_DIR = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "BCI_Competion4_dataset4_data_fingerflexions"
    )
    DEFAULT_PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed")
    DEFAULT_LOG_FUNC = print

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.save_dir = kwargs.pop('save_dir', self.DEFAULT_PREPROCESSED_DATA_DIR)
        self.raw_data_dir = kwargs.pop('raw_data_dir', self.DEFAULT_RAW_DATA_DIR)
        self.time_delay_secs = kwargs.pop('time_delay_secs', self.DEFAULT_TIME_DELAY_SECS)
        self.log_func = kwargs.pop('log_func', self.DEFAULT_LOG_FUNC)
        if self.log_func == "tqdm":
            self.log_func = self.push_log_to_p_bar
        self.p_bar = kwargs.pop('p_bar', None)
        self.ecog_train_data = None
        self.fingerflex_train_data = None
        self.ecog_test_data = None
        self.fingerflex_test_data = None

    def push_log_to_p_bar(self, msg):
        if self.p_bar is not None:
            self.p_bar.set_postfix_str(msg)
            self.p_bar.update()

    def interpolate_fingerflex(
            self,
            finger_flex,
            cur_fs=1000,
            true_fs=25,
            needed_hz=DOWNSAMPLE_FS,
            interp_type='cubic'
    ):
        """
        Interpolation of the finger motion recording to match the new sampling rate
        :param finger_flex: Initial sequences with finger flexions data
        :param cur_fs: ECoG sampling rate
        :param true_fs: Actual finger motions recording sampling rate
        :param needed_hz: Required sampling rate
        :param interp_type: Type of interpolation. By default - cubic
        :return: Returns an interpolated set of finger motions with the desired sampling rate
        """
        self.log_func("Interpolating fingerflex...")
        downscaling_ratio = cur_fs // true_fs
        self.log_func("Computing true_fs values...")
        finger_flex_true_fs = finger_flex[:, ::downscaling_ratio]
        finger_flex_true_fs = np.c_[finger_flex_true_fs,
        finger_flex_true_fs.T[-1]]  # Add as the last value on the interpolation edge the last recorded
        # Because otherwise it is not clear how to interpolate the tail at the end

        upscaling_ratio = needed_hz // true_fs

        ts = np.asarray(range(finger_flex_true_fs.shape[1])) * upscaling_ratio

        self.log_func("Making funcs...")
        interpolated_finger_flex_funcs = [scipy.interpolate.interp1d(ts, finger_flex_true_fs_ch, kind=interp_type) for
                                          finger_flex_true_fs_ch in finger_flex_true_fs]
        ts_needed_hz = np.asarray(range(finger_flex_true_fs.shape[1] * upscaling_ratio)[
                                  :-upscaling_ratio])  # Removing the extra added edge

        self.log_func("Interpolating with needed frequency")
        interpolated_finger_flex = np.array([[interpolated_finger_flex_func(t) for t in ts_needed_hz] for
                                             interpolated_finger_flex_func in interpolated_finger_flex_funcs])
        return interpolated_finger_flex

    def reshape_column_ecog_data(self, multichannel_signal: np.ndarray):
        return multichannel_signal.T  # (time, features) -> (features, time)

    def normalize(self, multichannel_signal: np.ndarray, return_values=None):
        """
        standardization and removal of the median  from each channel
        :param multichannel_signal: Multi-channel signal
        :param return_values: Whether to return standardization parameters. By default - no
        """
        self.log_func("Normalizing...")
        means = np.mean(multichannel_signal, axis=1, keepdims=True)
        stds = np.std(multichannel_signal, axis=1, keepdims=True)
        transformed_data = (multichannel_signal - means) / stds
        common_average = np.median(transformed_data, axis=0, keepdims=True)
        transformed_data = transformed_data - common_average
        if return_values:
            return transformed_data, (means, stds)
        self.log_func("Normalized.")
        return transformed_data

    def filter_ecog_data(self, multichannel_signal: np.ndarray, fs=1000, powerline_freq=50):
        """
        Harmonics removal and frequency filtering
        :param multichannel_signal: Initial multi-channel signal
        :param fs: Sampling rate
        :param powerline_freq: Grid frequency
        :return: Filtered signal
        """
        harmonics = np.array([i * powerline_freq for i in range(1, (fs // 2) // powerline_freq)])

        self.log_func("Starting...")
        signal_filtered = mne.filter.filter_data(
            multichannel_signal, fs, l_freq=self.L_FREQ, h_freq=self.H_FREQ
        )  # remove all frequencies between l and h
        self.log_func("Noise frequencies removed...")
        signal_removed_powerline_noise = mne.filter.notch_filter(
            signal_filtered, fs, freqs=harmonics
        )  # remove powerline  noise
        self.log_func("Powerline noise removed.")

        return signal_removed_powerline_noise

    def compute_spectrogramms(
            self,
            multichannel_signal: np.ndarray,
            fs=1000,
            freqs=np.logspace(np.log10(L_FREQ), np.log10(H_FREQ), WAVELET_NUM),
            output_type='power'
    ):
        """
        Compute spectrogramms using wavelet transforms

        :param freqs: wavelet frequencies to uses
        :param fs: Sampling rate
        :return: Signal spectogramms in shape (channels, wavelets, time)
        """

        num_of_channels = len(multichannel_signal)

        self.log_func("Computing wavelets...")
        spectrogramms = mne.time_frequency.tfr_array_morlet(
            multichannel_signal.reshape(1, num_of_channels, -1),
            sfreq=fs, freqs=freqs, output=output_type, verbose=10, n_jobs=6
        )[0]

        self.log_func("Wavelet spectrogramm computed.")

        return spectrogramms

    def downsample_spectrogramms(self, spectrogramms: np.ndarray, cur_fs=1000, needed_hz=H_FREQ, new_fs=None):
        """
        Reducing the sampling rate of spectrograms
        :param spectrogramms: Original set of spectrograms
        :param cur_fs: Current sampling rate
        :param needed_hz: The maximum frequency that must be unambiguously preserved during compression
        :param new_fs: The required sampling rate (interchangeable with needed_hz)
        :return: Decimated signal
        """
        self.log_func("Downsampling spectrogramm...")
        if new_fs == None:
            new_fs = needed_hz * 2
        downsampling_coef = cur_fs // new_fs
        assert downsampling_coef > 1
        downsampled_spectrogramm = spectrogramms[:, :, ::downsampling_coef]
        self.log_func("Spectrogramm downsampled.")
        return downsampled_spectrogramm

    def normalize_spectrogramms_to_db(self, spectrogramms: np.ndarray, convert=False):
        """
        Optional conversion to db, not used in the final version
        """
        if convert:
            return np.log10(spectrogramms + 1e-12)
        else:
            return spectrogramms

    def crop_for_time_delay(
            self,
            finger_flex: np.ndarray,
            spectrogramms: np.ndarray,
            time_delay_sec: float,
            fs: int
    ):
        """
        Taking into account the delay between brain waves and movements
        :param finger_flex: Finger flexions
        :param spectrogramms: Computed spectrogramms
        :param time_delay_sec: time delay hyperparameter
        :param fs: Sampling rate
        :return: Shifted series with a delay
        """

        time_delay = int(time_delay_sec * fs)

        # the first motions do not depend on available data
        finger_flex_cropped = finger_flex[..., time_delay:]
        # The latter spectrograms have no corresponding data
        spectrogramms_cropped = spectrogramms[..., :spectrogramms.shape[2] - time_delay]
        return finger_flex_cropped, spectrogramms_cropped

    def load_raw_data_and_preprocess(
            self,
            mat_path: str,
            data_key: str,
            dg_key: str,
    ):
        """
        Loading the raw data and applying the processing algorithm
        """
        data = scipy.io.loadmat(mat_path)

        interpolated_finger_flex = self.interpolate_fingerflex(
            finger_flex=self.reshape_column_ecog_data(data[dg_key].astype('float64'))
        )

        reshaped_train_data = self.reshape_column_ecog_data(data[data_key].astype('float64'))
        normalized_train_data, (means, stds) = self.normalize(reshaped_train_data, return_values=True)
        filtered_train_data = self.filter_ecog_data(normalized_train_data)
        spectrogramms = self.compute_spectrogramms(filtered_train_data)
        downsampled_spectrogramms = self.downsample_spectrogramms(spectrogramms, new_fs=self.DOWNSAMPLE_FS)
        db_spectrogramms = self.normalize_spectrogramms_to_db(spectrogramms=downsampled_spectrogramms)
        return interpolated_finger_flex, db_spectrogramms

    def load_raw_tr_data_and_preprocess(
            self,
            mat_path: Optional[str] = None,
            data_key: Optional[str] = None,
            dg_key: Optional[str] = None,
    ):
        """
        Loading the raw data and applying the processing algorithm
        """
        if mat_path is None:
            mat_path = os.path.join(self.raw_data_dir, self.TR_DEFAULT_MAT)
        if data_key is None:
            data_key = self.TR_DATA_KEY
        if dg_key is None:
            dg_key = self.TR_DG_KEY
        return self.load_raw_data_and_preprocess(mat_path, data_key, dg_key)

    def load_raw_test_data_and_preprocess(
            self,
            mat_path: Optional[str] = None,
            data_key: Optional[str] = None,
            dg_key: Optional[str] = None,
    ):
        """
        Loading the raw data and applying the processing algorithm
        """
        if mat_path is None:
            mat_path = os.path.join(self.raw_data_dir, self.TEST_DEFAULT_MAT)
        if data_key is None:
            data_key = self.TEST_DATA_KEY
        if dg_key is None:
            dg_key = self.TEST_DG_KEY
        return self.load_raw_data_and_preprocess(mat_path, data_key, dg_key)

    def download_raw_data(self, url: str = None, save_dir: str = None, **kwargs):
        """
        Downloading the raw data
        """
        import requests
        import zipfile
        import shutil

        url = url or self.RAW_DATA_URL
        save_dir = save_dir or self.raw_data_dir
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.log_func(f"Downloading raw data from {url} to {save_dir}")
        r = requests.get(url, stream=True)
        self.log_func(f"Saving raw data to {save_dir}")
        with open(os.path.join(save_dir, "raw_data.zip"), 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        with zipfile.ZipFile(os.path.join(save_dir, "raw_data.zip"), 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        self.log_func(f"Raw data saved to {save_dir}")

    def maybe_download_raw_data(self, url: str = None, save_dir: str = None, **kwargs):
        """
        Downloading the raw data if it is not already downloaded
        """
        if not os.path.exists(os.path.join(self.raw_data_dir, self.TR_DEFAULT_MAT)):
            self.download_raw_data(url, save_dir, **kwargs)

    def run(self, **kwargs):
        """
        Run the preprocessing pipeline
        """
        kwargs.update(self.kwargs)

        self.maybe_download_raw_data(**kwargs)

        current_fs = self.DOWNSAMPLE_FS
        interpolated_finger_flex_train, db_spectrogramms_train = self.load_raw_tr_data_and_preprocess()
        interpolated_finger_flex_test, db_spectrogramms_test = self.load_raw_test_data_and_preprocess()

        interpolated_finger_flex_train_cropped, db_spectrogramms_train_cropped = self.crop_for_time_delay(
            interpolated_finger_flex_train, db_spectrogramms_train, self.time_delay_secs, current_fs
        )
        interpolated_finger_flex_test_cropped, db_spectrogramms_test_cropped = self.crop_for_time_delay(
            interpolated_finger_flex_test, db_spectrogramms_test, self.time_delay_secs, current_fs
        )
        self.ecog_train_data = db_spectrogramms_train_cropped
        self.fingerflex_train_data = interpolated_finger_flex_train_cropped
        self.ecog_test_data = db_spectrogramms_test_cropped
        self.fingerflex_test_data = interpolated_finger_flex_test_cropped

        self.finger_motions_scaling(**kwargs)
        self.ecog_data_scaling(**kwargs)

        return self.save(**kwargs)

    def save(self, save_dir: Optional[str] = None, reshape=False, **kwargs):
        kwargs.update(self.kwargs)
        save_dir = save_dir or self.save_dir

        self.log_func(f"Saving preprocessed data to {save_dir}")

        pathlib.Path(f"{save_dir}/train").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{save_dir}/val").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{save_dir}/test").mkdir(parents=True, exist_ok=True)

        ecog_train_path = f"{save_dir}/train/ecog_data.npy"
        fingerflex_train_path = f"{save_dir}/train/fingerflex_data.npy"
        ecog_test_path = f"{save_dir}/test/ecog_data.npy"
        fingerflex_test_path = f"{save_dir}/test/fingerflex_data.npy"

        if reshape:
            self.ecog_train_data = self.ecog_train_data.reshape(self.CHANNELS_NUM * self.WAVELET_NUM, -1)
            self.ecog_test_data = self.ecog_test_data.reshape(self.CHANNELS_NUM * self.WAVELET_NUM, -1)

        np.save(ecog_train_path, self.ecog_train_data)
        np.save(fingerflex_train_path, self.fingerflex_train_data)
        np.save(ecog_test_path, self.ecog_test_data)
        np.save(fingerflex_test_path, self.fingerflex_test_data)

        self.log_func(f"Preprocessed data saved to {save_dir}")

        return ecog_train_path, fingerflex_train_path, ecog_test_path, fingerflex_test_path

    def load(self, load_dir: Optional[str] = None, **kwargs):
        kwargs.update(self.kwargs)
        load_dir = load_dir or self.save_dir
        ecog_train_path = f"{load_dir}/train/ecog_data.npy"
        fingerflex_train_path = f"{load_dir}/train/fingerflex_data.npy"
        ecog_test_path = f"{load_dir}/test/ecog_data.npy"
        fingerflex_test_path = f"{load_dir}/test/fingerflex_data.npy"

        self.log_func(f"Loading preprocessed data from {load_dir}")

        self.ecog_train_data = np.load(ecog_train_path)
        self.fingerflex_train_data = np.load(fingerflex_train_path)
        self.ecog_test_data = np.load(ecog_test_path)
        self.fingerflex_test_data = np.load(fingerflex_test_path)

        self.log_func(f"Preprocessed data loaded from {load_dir}")

        return ecog_train_path, fingerflex_train_path, ecog_test_path, fingerflex_test_path

    def finger_motions_scaling(self, **kwargs):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaler.fit(self.fingerflex_train_data.T)

        self.fingerflex_train_data = scaler.transform(self.fingerflex_train_data.T).T
        self.fingerflex_test_data = scaler.transform(self.fingerflex_test_data.T).T
        return self.save(**kwargs)

    def ecog_data_scaling(self, **kwargs):
        from sklearn.preprocessing import RobustScaler

        transformer = RobustScaler(unit_variance=True, quantile_range=(0.1, 0.9))
        transformer.fit(self.ecog_train_data.T.reshape(-1, self.WAVELET_NUM * self.CHANNELS_NUM))

        self.ecog_train_data = transformer.transform(
            self.ecog_train_data.T.reshape(-1, self.WAVELET_NUM * self.CHANNELS_NUM)
        ).reshape(-1, self.WAVELET_NUM, self.CHANNELS_NUM).T
        self.ecog_test_data = transformer.transform(
            self.ecog_test_data.T.reshape(-1, self.WAVELET_NUM * self.CHANNELS_NUM)
        ).reshape(-1, self.WAVELET_NUM, self.CHANNELS_NUM).T
        return self.save(**kwargs)
