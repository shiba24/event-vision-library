"""Convert existing AEDAT4 format event file into hdf5 file."""

from evlib.codec import fileformat


# Setup iterator reader
aedat4_file_path = "./artifacts/sample_data/sample.aedat4"
ev_iter = fileformat.IteratorAedat4Event(aedat4_file_path)

# Define key mapping for the hdf5 file
data_keys = {"x": "raw_events/x", "y": "raw_events/y", "t": "raw_events/t", "p": "raw_events/p"}

# Convert
output_file_name = "event.hdf5"
fileformat.convert_iterator_access_to_hdf5(ev_iter, output_file_name, data_keys)
print(f"Finished conversion: {output_file_name}")
