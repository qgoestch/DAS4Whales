# Download directly from the OOI DAS experiment - details here:
# https://oceanobservatories.org/pi-instrument/ \
# rapid-a-community-test-of-distributed-acoustic-sensing-on-the-ocean-observatories-initiative-regional-cabled-array/

import os

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import wget

import das4whales as dw


def main(url: str):

    filepath, filename = dw.data_handle.dl_file(url)

    # ### Get information on the DAS data from the hdf5 metadata

    # Read HDF5 files and access metadata
    # Get the acquisition parameters for the data folder
    metadata = dw.data_handle.get_acquisition_parameters(
        filepath, interrogator="optasense"
    )

    fs, dx, nx, ns, gauge_length, scale_factor = (
        metadata["fs"],
        metadata["dx"],
        metadata["nx"],
        metadata["ns"],
        metadata["GL"],
        metadata["scale_factor"],
    )
    # Select desired channels
    selected_channels_m = [
        20000,
        65000,
        10,
    ]  # [20000, 65000, 10]  # list of values in meters corresponding to the starting,
    # ending and step wanted channels along the FO Cable
    # selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
    # in meters
    selected_channels = [
        int(np.floor(selected_channels_m / dx))
        for selected_channels_m in selected_channels_m
    ]  # list of values in channel number (spatial sample) corresponding
    # to the starting, ending and step wanted
    # channels along the FO Cable
    # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
    # numbers
    # Create conditioning for the signal

    # Create high-pass filter
    sos_hpfilt = dw.dsp.butterworth_filter([2, 5, "hp"], fs)

    # Create band-pass filter for the TX plots
    sos_bpfilt = dw.dsp.butterworth_filter([5, [10, 30], "bp"], fs)

    # Load DAS data
    tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(
        filepath, selected_channels, metadata, interrogator="optasense"
    )

    # apply the high-pass filter
    trf = sp.sosfiltfilt(sos_hpfilt, tr, axis=1)

    fk_params = {
        "fmin": 10,
        "fmax": 30,
        "c_min": 1400,
        "c_max": 3500,
    }

    # FK filter
    # Create the f-k filter
    fk_filter = dw.dsp.hybrid_filter_design(
        (trf.shape[0], trf.shape[1]),
        selected_channels,
        dx,
        fs,
        fk_params=fk_params,
        display_filter=True
    )

    # Apply the f-k filter to the data
    trf_fk = dw.dsp.fk_filter_filt(trf, fk_filter)

    # Spatio-temporal plot high-pass filtered data
    dw.plot.plot_tx(
        trf, time, dist, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=0.2
    )
    plt.show()

    # Spatio-temporal plot high-pass + f-k filtered data
    dw.plot.plot_tx(
        trf_fk, time, dist, fileBeginTimeUTC, fig_size=(12, 10), v_min=0, v_max=0.2
    )
    plt.show()

    # Spatio-spectral plot
    dw.plot.plot_fx(
        trf_fk,
        dist,
        fs,
        title_time_info=fileBeginTimeUTC,
        win_s=2,
        nfft=512,
        f_min=10,
        f_max=35,
        fig_size=(25, 10),
        v_min=0,
        v_max=0.08,
    )


if __name__ == "__main__":
    url = (
        "http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/"
        "North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/"
        "North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5"
    )
    main(url)
