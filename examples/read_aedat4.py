"""Read an AEDAT4 file and print a summary of each of its streams."""

from evlib.codec import fileformat


aedat4_file_path = "./artifacts/sample_data/sample.aedat4"

# Events
# Each batch is a RawEvents object. RawEvents.as_numpy() returns [y, x, t, p].
event_iterator = fileformat.IteratorAedat4Event(aedat4_file_path)
num_events = 0
for events in event_iterator:
    num_events += len(events)
print(f"events  : {num_events}")

# Frames
# Each packet is a dict with the image array: "frame", timestamp "t", and "num"
frame_iterator = fileformat.IteratorAedat4Frame(aedat4_file_path)
num_frames = 0
for frame in frame_iterator:
    num_frames += frame["num"]
print(f"frames  : {num_frames}")

# IMU
# Each packet is a dict with sample array "imu" and its count "num".
imu_iterator = fileformat.IteratorAedat4Imu(aedat4_file_path)
num_imu = 0
for imu in imu_iterator:
    num_imu += imu["num"]
print(f"imu     : {num_imu}")

# Triggers
# Each packet is a dict with trigger array "trigger" and its count "num".
trigger_iterator = fileformat.IteratorAedat4Trigger(aedat4_file_path)
num_triggers = 0
for trigger in trigger_iterator:
    num_triggers += trigger["num"]
print(f"triggers: {num_triggers}")
