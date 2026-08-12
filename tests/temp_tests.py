import das4whales.data_handle as dh
import time

# filepath = r"C:\Users\ers334\Desktop\testingData\OOI\DASData\OptaSense\North_C2\North-C2-HF-P1kHz-GL30m-Sp2m-FS500Hz_2021-11-02T215901Z.h5"
# interrogator = 'optasense'

# filepath = r"C:\Users\ers334\Desktop\testingData\Svalbard\data\120117.hdf5"
# interrogator = 'asn'

filepath = r"C:\Users\ers334\Desktop\testingData\OOI\DASData\Silixa_DAS_South90km\OOIPacCity_UTC_20211104_180903.573.tdms"
interrogator = 'silixa'

tic = time.time()
metadata = dh.get_acquisition_parameters(filepath=filepath, interrogator = interrogator)
print(f"Time to load metadata: {time.time() - tic} seconds")

# trace, tx, dist, file_begin_time_utc = dh.load_das_data(filepath, selected_channels = [int(1e3), int(11e3), int(4)], metadata = metadata, interrogator = interrogator)
# print(trace.shape)