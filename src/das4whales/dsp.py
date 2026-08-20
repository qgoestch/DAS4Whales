"""
dsp.py - Digital Signal Processing module for DAS4Whales

This module provides various functions for digital signal processing of DAS strain data.

Authors: Léa Bouffaut, Quentin Goestchel
Date: 2023-2024-2025
"""

import cv2
import deprecation
import librosa
import numpy as np
import scipy.fft as sfft
import scipy.signal as sp
import sparse
from numpy.fft import fftfreq, fftshift, ifft2, ifftshift
from scipy import ndimage


# Digital sampling
def resample(
    tr: np.ndarray, fs: int, desired_fs: int
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Resample a multi-channel signal to a desired sampling frequency.

    Parameters
    ----------
    tr : ndarray
        Input signal with shape `(n_channels, n_samples)`, where `n_channels` is the number of channels
        and `n_samples` is the number of time samples.
    fs : int
        Original sampling frequency of the input signal (in Hz).
    desired_fs : int
        Desired sampling frequency after resampling (in Hz).

    Returns
    -------
    tuple
        A tuple containing:
        - tr_downsampled (ndarray): Downsampled signal with shape `(n_channels, n_samples_downsampled)`.
        - fs_downsampled (int): Sampling frequency after downsampling (equals `desired_fs`).
        - tx_downsampled (ndarray): New time vector corresponding to the downsampled signal,
          with shape `(n_samples_downsampled,)`.
    """
    # 1) Filter the signal if downsampling is needed
    if desired_fs < fs:
        # Butterworth low-pass filter
        sos = sp.butter(8, desired_fs / 2, "low", fs=fs, output="sos")
        tr = sp.sosfiltfilt(sos, tr, axis=-1)

    # 2) Resample
    tr_downsampled = librosa.resample(
        tr, axis=1, orig_sr=fs, target_sr=desired_fs, res_type="soxr_vhq"
    )  # axis specifies the time axis
    fs_downsampled = desired_fs
    n_channels, n_samples = tr_downsampled.shape

    # 3) New time vector
    tx_downsampled = np.arange(n_samples) / fs_downsampled

    return tr_downsampled, fs_downsampled, tx_downsampled


# Transformations
def get_fx(trace: np.ndarray, nfft: int) -> np.ndarray:
    """
    Apply a fast Fourier transform (FFT) to each channel of the strain data matrix.

    Parameters
    ----------
    trace : np.ndarray
        A 2D array of shape (channel, time sample) containing the strain data in the spatio-temporal domain.
    nfft : int
        Number of time samples used for the FFT.

    Returns
    -------
    ndarray
        A 2D array of shape (channel, freq. sample) containing the strain data in the spatio-spectral domain.
    """

    fx = 2 * (abs(np.fft.fftshift(np.fft.fft(trace, nfft), axes=1)))
    fx /= nfft
    fx *= 10**9
    return fx


def get_spectrogram(
    waveform: np.ndarray, fs: float, nfft: int = 128, overlap_pct: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the spectrogram of a single channel

    Parameters
    ----------
    waveform : np.ndarray
        Single channel temporal signal.
    fs : float
        The sampling frequency (Hz).
    nfft : int, optional
        Number of time samples used for the STFT. Default is 128.
    overlap_pct : float, optional
        Percentage of overlap in the spectrogram. Default is 0.8.

    Returns
    -------
    p : np.ndarray
        Spectrogram in dB scale (normalized by max).
    tt : np.ndarray
        Time vector.
    ff : ndarray
        Frequency vector.

    """
    spectrogram = np.abs(
        librosa.stft(
            y=waveform, n_fft=nfft, hop_length=int(np.floor(nfft * (1 - overlap_pct)))
        )
    )

    # Axis
    height = spectrogram.shape[0]
    width = spectrogram.shape[1]

    tt = np.linspace(0, len(waveform) / fs, num=width)
    ff = np.linspace(0, fs / 2, num=height)
    p = 20 * np.log10(spectrogram / np.max(spectrogram))

    return p, tt, ff


def normalize_std(trace: np.ndarray) -> np.ndarray:
    """
    Normalize the input trace by its standard deviation.

    Parameters
    ----------
    trace : np.ndarray
        A 2D array of shape (channel, time sample) containing the strain data in the spatio-temporal domain.

    Returns
    -------
    np.ndarray
        A 2D array of shape (channel, time sample) containing the normalized strain data.
    """

    return trace / np.std(trace, axis=1, keepdims=True)


def normalize_median(trace: np.ndarray) -> np.ndarray:
    """
    Normalize the input trace by its median.

    Parameters
    ----------
    trace : np.ndarray
        A 2D array of shape (channel, time sample) containing the strain data in the spatio-temporal domain.

    Returns
    -------
    np.ndarray
        A 2D array of shape (channel, time sample) containing the normalized strain data.
    """

    return trace / np.median(trace, axis=1, keepdims=True)


# Filters
# f-k filters design functions


def hybrid_filter_design(
    trace_shape: tuple[int, int],
    selected_channels: list[int],
    dx: float,
    fs: float,
    fk_params: dict[str, float],
    inf_wspeed: bool = False,
    taper: str = "gaussian",
    display_filter: bool = False,
) -> sparse.COO:
    """Designs a bandpass f-k hybrid filter for DAS strain data. a.k.a the "butterfly" filter.

    Universal filter for frequency-wavenumber domain processing with flexible
    wave speed and tapering constraints. Default configuration optimized for
    fin whale detection with non-infinite wave speed bounds and gaussian tapering.

    Parameters
    ----------
    trace_shape : tuple[int, int]
        Tuple with the dimensions of the strain data [channels, samples].
    selected_channels : list[int]
        List of selected channel indices [start, end, step].
    dx : float
        Channel spacing (m).
    fs : float
        Sampling frequency (Hz).
    fk_params : dict
        Dictionary containing filter parameters:
        - 'fmin': minimum frequency for passband (Hz)
        - 'fmax': maximum frequency for passband (Hz)
        - 'c_min': minimum phase speed for passband (m/s) [ignored if infinite_wave_speed=True]
        - 'c_max': maximum phase speed for passband (m/s) [ignored if infinite_wave_speed=True]
        - 'df_taper': frequency transition width (Hz), default 4
        - 'speed_taper': relative speed transition width, default 0.05
    inf_wspeed : bool, optional
        If True, filter passes all wavenumbers in the frequency band (no speed constraints).
        If False (default), constrains pass band to speeds between c_min and c_max.
    taper : str, optional
        Type of taper to apply for smooth transitions. Options:
        - 'gaussian' (default): Gaussian blur with sigma=40
        - 'sine': Sine taper window
    display_filter : bool, optional
        Whether to display filter visualization. Default is False.

    Returns
    -------
    sparse.COO
        Sparse f-k filter matrix with shape [channels, samples].

    Raises
    ------
    ValueError
        If taper is not 'gaussian' or 'sine'.

    See Also
    --------
    fk_filter_filt : Apply the f-k filter to DAS data
    """
    # Validate taper parameter
    if taper not in ("gaussian", "sine"):
        raise ValueError(f"taper must be 'gaussian' or 'sine', got '{taper}'")
    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx
    # Get the dimensions of the trace data
    nnx, nns = trace_shape
    fmin = fk_params["fmin"]
    fmax = fk_params["fmax"]
    if inf_wspeed:
        c_min = fk_params.get("c_min", np.inf)
        c_max = fk_params.get("c_max", np.inf)
    else:
        c_min = fk_params["c_min"]
        c_max = fk_params["c_max"]

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=selected_channels[2] * dx))

    # Find the corresponding indexes of the frequencies of interest
    fmin_idx = np.argmax(freq >= fmin)
    fmax_idx = np.argmax(freq >= fmax)

    # Sine tapering from older version of the code, kept for reference
    if taper == "sine":
        # Build frequency transitions around the passband.
        df_taper = fk_params.get("df_taper", 4.0)
        speed_taper = fk_params.get("speed_taper", 0.1)
        if df_taper <= 0 or speed_taper < 0 or speed_taper >= 1:
            raise ValueError(
                "df_taper must be positive and speed_taper must be in [0, 1)"
            )

        fpmin = fmin - df_taper
        fpmax = fmax + df_taper
        if fpmin < 0:
            raise ValueError("df_taper must define valid frequency transitions")

        fk_filter_matrix = np.zeros((len(knum), len(freq)))
        rising_mask = (freq >= fpmin) & (freq <= fmin)
        falling_mask = (freq >= fmax) & (freq <= fpmax)
        frequency_response = np.zeros_like(freq)
        frequency_response[rising_mask] = np.sin(
            0.5 * np.pi * (freq[rising_mask] - fpmin) / df_taper
        )
        frequency_response[(freq >= fmin) & (freq <= fmax)] = 1
        frequency_response[falling_mask] = np.cos(
            0.5 * np.pi * (freq[falling_mask] - fmax) / df_taper
        )
        fk_filter_matrix = np.tile(frequency_response, (len(knum), 1))

        # Apply speed transitions inferred around the passband limits.
        for i in range(np.argmax(freq >= fpmin), np.argmax(freq >= fpmax)):
            if inf_wspeed:
                filter_col = (knum > 0) & (knum < freq[i] / c_min)
            else:
                cs_min = c_min * (1 - speed_taper)
                cs_max = c_max * (1 + speed_taper)
                kp_min = freq[i] / c_max
                kp_max = freq[i] / c_min
                ks_min = freq[i] / cs_max
                ks_max = freq[i] / cs_min
                filter_col = np.zeros_like(knum)
                filter_col[(knum > kp_min) & (knum < kp_max)] = 1
                lower_mask = (knum >= ks_min) & (knum <= kp_min)
                upper_mask = (knum >= kp_max) & (knum <= ks_max)
                filter_col[lower_mask] = np.sin(
                    0.5 * np.pi * (knum[lower_mask] - ks_min) / (kp_min - ks_min)
                )
                filter_col[upper_mask] = np.sin(
                    0.5 * np.pi * (ks_max - knum[upper_mask]) / (ks_max - kp_max)
                )
            fk_filter_matrix[:, i] *= filter_col

        fk_filter_matrix += np.fliplr(fk_filter_matrix)
        fk_filter_matrix += np.flipud(fk_filter_matrix)
    # Gaussian tapering for smooth transitions
    else:
        # Initiate the hard-edged filter before applying Gaussian smoothing.
        fk_filter_matrix = np.zeros((len(knum), len(freq)))
        for i in range(fmin_idx, fmax_idx):
            filter_col = np.zeros_like(knum)
            if inf_wspeed:
                filter_col[(knum < freq[i] / c_min) & (knum > 0)] = 1
            else:
                kp_min = freq[i] / c_max
                kp_max = freq[i] / c_min
                filter_col[(knum > kp_min) & (knum < kp_max)] = 1
            fk_filter_matrix[:, i] = filter_col

        # Apply Gaussian blur for smooth transitions
        sub_matrix = fk_filter_matrix[
            len(knum) // 2 : len(knum), len(freq) // 2 : len(freq)
        ].astype(np.float32)
        tapered_sub_matrix = cv2.GaussianBlur(sub_matrix, (0, 0), 40)
        fk_filter_matrix[len(knum) // 2 :, len(freq) // 2 :] = tapered_sub_matrix
        fk_filter_matrix += np.fliplr(fk_filter_matrix)
        fk_filter_matrix += np.flipud(fk_filter_matrix)

    # Filter display, optional
    if display_filter:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        # Context manager for the plot (to avoid changing the global settings)
        with plt.rc_context():
            fig = plt.figure(figsize=(14.8, 8.8))
            gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[6, 2])

            # Matrix display
            ax1 = plt.subplot(gs[0])
            ax1.imshow(
                fk_filter_matrix,
                extent=[min(freq), max(freq), min(knum), max(knum)],
                aspect="auto",
                origin="lower",
            )
            ax1.hlines(
                knum[len(knum) // 2 + 420],
                min(freq),
                max(freq),
                color="tab:orange",
                lw=4,
                ls=":",
            )
            ax1.vlines(
                freq[len(freq) // 2 + 1500],
                min(knum),
                max(knum),
                color="tab:blue",
                lw=4,
                ls=":",
            )
            ax1.set_ylabel("Wavenumber [m$^{-1}$]")

            # Frequency slice display
            ax2 = plt.subplot(gs[2], sharex=ax1)
            ax2.plot(
                freq,
                fk_filter_matrix[len(knum) // 2 + 420, :],
                lw=3,
                color="tab:orange",
            )
            ax2.set_xlabel("Frequency [Hz]")
            ax2.set_ylabel("Gain []")
            ax2.set_xlim([min(freq), max(freq)])
            ax2.grid()

            # Wavenumber slice display
            ax3 = plt.subplot(gs[1], sharey=ax1)
            ax3.plot(
                fk_filter_matrix[:, len(freq) // 2 + 1500], knum, lw=3, color="tab:blue"
            )
            ax3.set_xlabel("Gain []")
            ax3.set_ylabel("Wavenumber [m$^{-1}$]")
            ax3.yaxis.set_label_position("right")
            ax3.set_ylim([min(knum), max(knum)])
            ax3.invert_xaxis()
            ax3.yaxis.tick_right()
            ax3.grid()
            plt.tight_layout()
            plt.show()

    return sparse.COO.from_numpy(fk_filter_matrix)


def taper_data(trace: np.ndarray) -> np.ndarray:
    """
    Apply a Tukey window to each line (time series) of the input matrix.

    Parameters
    ----------
    trace : np.ndarray
        2D numpy array, where each column represents a time series.

    Returns
    -------
    np.ndarray
        Tapered matrix with the same shape as the input.
    """
    nt = trace.shape[1]
    # Change alpha to increase the tapering ratio
    trace *= sp.windows.tukey(nt, alpha=0.03)[np.newaxis, :]
    return trace


def taper_data2d(data: np.ndarray, taper_type: str = "tukey") -> np.ndarray:
    """
    Applies tapering (windowing) to the data in both space (channels) and time (samples) domains.

    Parameters
    ----------
    data : ndarray
        2D numpy array representing the spatio-temporal data with dimensions [channels x samples].
    taper_type : str, optional
        Type of tapering window to apply. Options are 'hanning', 'hamming', or 'tukey'. Default is 'tukey'.

    Returns
    -------
    tapered_data : ndarray
        The tapered data array with the same shape as input data.
    """

    # Get the shape of the data
    n_channels, n_samples = data.shape

    # Choose the taper window for space (channels) and time (samples)
    if taper_type == "hanning":
        spatial_taper = np.hanning(n_channels)  # Hanning window for spatial dimension
        temporal_taper = np.hanning(n_samples)  # Hanning window for temporal dimension
    elif taper_type == "hamming":
        spatial_taper = np.hamming(n_channels)  # Hamming window for spatial dimension
        temporal_taper = np.hamming(n_samples)  # Hamming window for temporal dimension
    elif taper_type == "tukey":
        spatial_taper = sp.windows.tukey(
            n_channels, alpha=0.03
        )  # Tukey window for spatial dimension
        temporal_taper = sp.windows.tukey(
            n_samples, alpha=0.03
        )  # Tukey window for temporal dimension
    else:
        raise ValueError(
            "Unsupported taper_type. Choose 'hanning', 'hamming', or 'tukey'."
        )

    # Apply the tapers
    tapered_data = data * spatial_taper[:, np.newaxis] * temporal_taper[np.newaxis, :]

    return tapered_data


def fk_filter_filt(
    trace: np.ndarray,
    fk_filter_matrix: np.ndarray | sparse.COO,
    tapering: bool = False,
    parallel: int | None = None,
) -> np.ndarray:
    """Apply a pre-calculated f-k filter to DAS strain data.

    Parameters
    ----------
    trace : np.ndarray
        DAS data with shape ``[channels, samples]``.
    fk_filter_matrix : np.ndarray or sparse.COO
        Pre-calculated f-k filter with the same shape as ``trace``.
    tapering : bool, optional
        Whether to apply ``taper_data`` before filtering. Default is False.
    parallel : int or None, optional
        Number of workers for SciPy FFT operations. ``None`` uses NumPy's
        serial FFT implementation. Use ``-1`` to use all available workers.

    Returns
    -------
    np.ndarray
        Filtered DAS data with the same shape as ``trace``.
    """
    if tapering:
        trace = taper_data(trace)

    if parallel is not None:
        trace = np.asarray(trace, dtype=np.complex64)
        fk_trace = sfft.fft2(trace, workers=parallel)
    else:
        fk_trace = np.fft.fft2(trace)

    fk_filtered_trace = np.fft.fftshift(fk_trace) * fk_filter_matrix
    if isinstance(fk_filtered_trace, sparse.COO):
        fk_filtered_trace = fk_filtered_trace.todense()

    if parallel is not None:
        filtered_trace = sfft.ifft2(
            np.fft.ifftshift(fk_filtered_trace), workers=parallel
        )
    else:
        filtered_trace = np.fft.ifft2(np.fft.ifftshift(fk_filtered_trace))

    return filtered_trace.real


def butterworth_filter(
    filterspec: tuple[int, float | list[float], str], fs: float
) -> np.ndarray:
    """
    Designs and applies a Butterworth filter.

    Parameters:
    ----------
    filterspec : tuple
        A tuple containing the filter order, critical frequency, and filter type.
    fs : float
        The sampling frequency.

    Returns:
    -------
    filter_sos : np.ndarray
        The second-order sections (SOS) representation of the Butterworth filter.

    Notes:
    ------
    The Butterworth filter is designed using the scipy.signal.butter function.

    Example:
    --------
    filter_order = 4
    filter_critical_freq = 1000
    filter_type_str = 'lowpass'
    filterspec = (filter_order, filter_critical_freq, filter_type_str)
    fs = 44100

    filter_sos = butterworth_filter(filterspec, fs)
    trace_filtered = sp.sosfiltfilt(filter_sos, trace_original, axis=1)
    """

    filter_order, filter_critical_freq, filter_type_str = filterspec
    # Build a filter of the desired type
    wn = np.array(filter_critical_freq) / (fs / 2)  # convert to angular frequency

    filter_sos = sp.butter(filter_order, wn, btype=filter_type_str, output="sos")

    return filter_sos


def instant_freq(channel: np.ndarray, fs: float) -> np.ndarray:
    """Compute the instantaneous frequency

    Parameters
    ----------
    channel : np.ndarray
        1D time series channel trace
    fs : float
        sampling frequency

    Returns
    -------
    np.ndarray
        instantaneous frequency along time[1:]
    """
    # Compute the instantaneous frequency
    fi = np.diff(np.unwrap(np.angle(sp.hilbert(channel)))) / (2.0 * np.pi) * fs
    # Sliding window filtering to smooth out the fi
    # window_size = 50
    # ffi = np.convolve(fi, np.ones(window_size)/window_size, mode='same')
    # Compute the instantaneous median frequency
    # f, t, Zxx = sp.spectrogram(channel, fs, nperseg=140, noverlap=0.99)
    # cumulative_sum = np.cumsum(Zxx, axis=0)
    # Find the index corresponding to the median frequency at each time point
    # median_index = np.argmax(cumulative_sum >= 0.5 * cumulative_sum[-1], axis=0)
    # fm = f[median_index]
    return fi  # , ffi, t, fm


def bp_filt(data: np.ndarray, fs: float, fmin: float, fmax: float) -> np.ndarray:
    """bp_filt - perform bandpass filtering on an array of DAS data

    Parameters
    ----------
    data : array-like
        array containing wave signal from DAS data
    fs : float
        sampling frequency
    fmin : float
        minimum frequency for the passband
    fmax : float
        maximum frequency for the passband

    Returns
    -------
    tr_filt : array-like
        bandpass filtered data
    """
    b, a = sp.butter(8, [fmin / (fs / 2), fmax / (fs / 2)], "bp")
    tr_filt = sp.filtfilt(b, a, data, axis=1)
    return tr_filt


def fk_filt(
    data: np.ndarray,
    tint: float,
    fs: float,
    xint: float,
    dx: float,
    c_min: float,
    c_max: float,
    display_filter: bool = False,
) -> np.ndarray:
    """fk_filt - perform fk filtering on an array of DAS data

    Parameters
    ----------
    data : array-like
        array containing wave signal from DAS data
    tint : float
        decimation time interval between considered samples
    fs : float
        sampling frequency
    xint : float
        decimation space interval between considered samples
    dx : float
        spatial resolution
    c_min : float
        minimum phase speed for the pass-band filter in f-k domain
    c_max : float
        maximum phase speed for the pass-band filter in f-k domain

    Returns
    -------
    f : array-like
        vector of frequencies

    k : array-like
        vector of wavenumbers
    g : array-like
        2D designed gaussian filter
    data_fft_g: array-like
        2D Fourier transformed data, filtered by g
    data_g.real: array-like
        Real value of spatiotemporal filtered data
    """

    # Perform 2D Fourier Transform on the detrended input data
    data_fft = np.fft.fft2(data)
    # Make freq and wavenum vectors
    nx = data_fft.shape[0]
    ns = data_fft.shape[1]
    f = fftshift(fftfreq(ns, d=tint / fs))
    k = fftshift(fftfreq(nx, d=xint * dx))
    ff, kk = np.meshgrid(f, k)

    #  Define a filter in the f-k domain
    # Soundwaves have f/k = c so f = k*c

    g = 1.0 * ((ff < kk * c_min) & (ff < -kk * c_min))
    g2 = 1.0 * ((ff < kk * c_max) & (ff < -kk * c_max))

    # Symmetrize the filter
    g += np.fliplr(g)
    # g2 += np.fliplr(g2)
    g -= g2 + np.fliplr(g2)  # combine to have g = g - g2

    # Apply Gaussian filter to the f-k filter
    # Tuning the standard deviation of the filter can improve computational efficiency
    # Use a gaussian filter from openCV
    g = cv2.GaussianBlur(g, (0, 0), 60)
    # g = ndimage.gaussian_filter(g, 20)
    # epsilon = 0.0001
    # g = np.exp (-epsilon*( ff-kk*c)**2 )

    # Normalize the filter to values between 0 and 1
    g = (g - np.min(g)) / (np.max(g) - np.min(g))

    # Apply the filter to the 2D Fourier-transformed data
    data_fft_g = fftshift(data_fft) * g
    # Perform inverse Fourier Transform to obtain the filtered data in t-x domain
    data_g = ifft2(ifftshift(data_fft_g))

    # return f, k, g, data_fft_g, data_g.real

    # Filter display, optional
    if display_filter:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        # Context manager for the plot (to avoid changing the global settings)
        with plt.rc_context():
            # Change the font sizes for plots (if needed)
            # plt.rc('font', size=20)
            # plt.rc('xtick', labelsize=16)
            # plt.rc('ytick', labelsize=16)

            fig = plt.figure(figsize=(18, 10))
            gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[6, 2])

            # Matrix display
            ax1 = plt.subplot(gs[0])
            ax1.imshow(
                g,
                extent=[min(f), max(f), min(k), max(k)],
                aspect="auto",
                origin="lower",
            )
            ax1.hlines(
                k[len(k) // 2 + 420], min(f), max(f), color="tab:orange", lw=2, ls=":"
            )
            ax1.vlines(
                f[len(f) // 2 + 1500], min(k), max(k), color="tab:blue", lw=2, ls=":"
            )
            # colorbar
            # cbar = plt.colorbar(ax1.imshow(fk_filter_matrix, extent=[min(freq), max(freq), min(knum), max(knum)], aspect='auto', origin='lower'))
            ax1.set_ylabel("k [m$^{-1}$]")
            ax1.set_xlabel("f [Hz]")

            # Frequency slice display
            ax2 = plt.subplot(gs[2], sharex=ax1)
            ax2.plot(f, g[len(k) // 2 + 420, :], lw=3, color="tab:orange")
            ax2.set_xlabel("f [Hz]")
            ax2.set_ylabel("Gain []")
            ax2.set_xlim([min(f), max(f)])
            ax2.grid()

            # Wavenumber slice display
            ax3 = plt.subplot(gs[1], sharey=ax1)
            ax3.plot(g[:, len(f) // 2 + 1500], k, lw=3, color="tab:blue")
            ax3.set_xlabel("Gain []")
            ax3.set_ylabel("k [m$^{-1}$]")
            ax3.yaxis.set_label_position("right")
            ax3.set_ylim([min(k), max(k)])
            ax3.invert_xaxis()
            ax3.yaxis.tick_right()
            ax3.grid()
            plt.tight_layout()
            plt.show()

    return data_g.real


def snr_tr_array(trace: np.ndarray) -> np.ndarray:
    """Calculate the 2D Signal-to-Noise Ratio (SNR) array for a given input trace.

    This function computes the SNR for each element in the input 2D trace array. The SNR
    is calculated as the ratio of the square of the trace values to the square of the
    standard deviation of the trace along the second axis (time).

    Parameters
    ----------
    trace : numpy.ndarray
        The input 2D trace array for which the SNR is to be calculated.

    Returns
    -------
    numpy.ndarray
        A 2D array containing the Signal-to-Noise Ratio (SNR) values for each element
        in the input trace.
    """
    return 10 * np.log10(
        abs(sp.hilbert(trace, axis=1)) ** 2 / np.std(trace, axis=1, keepdims=True) ** 2
    )


def calc_snr_median(trace: np.ndarray) -> np.ndarray:
    """Calculate the Signal-to-Noise Ratio (SNR) for a given input trace.

    This function computes the SNR for the input trace. The SNR is calculated as the ratio of the square of the envelope of the trace to the square of the median of the trace.

    Parameters
    ----------
    trace : np.ndarray
        The input trace for which the SNR is to be calculated.

    Returns
    -------
    np.ndarray
        The Signal-to-Noise Ratio (SNR) value for the input trace.
    """

    envelope = abs(sp.hilbert(trace, axis=1))
    return 10 * np.log10(envelope**2 / np.median(envelope, axis=1, keepdims=True) ** 2)


def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(signal, np.ones(window_size) / window_size, mode="same")


def moving_average_matrix(matrix: np.ndarray, window_size: int) -> np.ndarray:
    return np.array(
        [moving_average(matrix[i, :], window_size) for i in range(matrix.shape[0])]
    )
