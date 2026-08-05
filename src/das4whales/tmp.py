import das4whales.data_handle as dh
import numpy as np
import matplotlib.pyplot as plt

filepath = r"C:\Users\ers334\Desktop\testingData\OOI\DASData\Silixa_DAS_South90km\OOIPacCity_UTC_20211104_180918.573.tdms"
# filepath = r"C:\Users\ers334\Desktop\testingData\OOI\DASData\OptaSense\North_C3\North-C3-HF-P1kHz-GL30m-Sp2m_2021-11-02T000345Z.h5"

metadata = dh.get_acquisition_parameters(filepath, interrogator="silixa")

trace, tx, dist, file_begin_time_utc = dh.load_das_data(filepath, interrogator="silixa", metadata=metadata, selected_channels = (0, 10000, 4))

print(trace.shape)