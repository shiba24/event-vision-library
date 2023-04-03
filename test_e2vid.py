import numpy as np
import cv2

from src.evlib.processing.reconstruction import E2Vid
from src.evlib.representation import Histogram
from src.evlib.processing.reconstruction.e2vid.utils.event_readers import FixedDurationEventReader


# Simple codre. There may be more efficient ways.
def extract_data(filename):
    infile = open(filename, 'r')
    timestamp = []
    x = []
    y = []
    pol = []
    for line in infile:
        words = line.split()
        timestamp.append(float(words[0]))
        x.append(int(words[1]))
        y.append(int(words[2]))
        pol.append(int(words[3]))
    infile.close()
    return timestamp,x,y,pol


if __name__ == '__main__':
    filename_sub = 'artifacts/sample_data/events_chunk.txt'
    # Call the function to read data    
    timestamp, x, y, pol = extract_data(filename_sub)
    img_size = (180,240)
    events = np.array([y, x, timestamp, pol]).T

    img = np.full(shape=img_size + (3,), fill_value=255, dtype="uint8")
    img[y, x, :] = 0
    img[y, x, pol] = 255
    cv2.imwrite(f"output/img.png", img)

    hist = Histogram(img_size, use_polarity=False)(events)
    cv2.imwrite(f"output/hist.png", (hist - np.min(hist)) / (np.max(hist) - np.min(hist)))

    reconstructor = E2Vid(img_size)

    loader = FixedDurationEventReader("artifacts/sample_data/dynamic_6dof.zip",
                                      duration_ms=33.33,
                                      start_index=0)
    new_column_order = [2, 1, 0, 3]

    num_events = 10000
    count = 0
    num_iterations = len(events) // num_events

    for i, events in enumerate(loader):
        if i == 10:
            break

        if events.shape[0] == 0:
            continue

        new_events = events[:, new_column_order]

        img = reconstructor(new_events)
        cv2.imwrite(f"output/example_{i}.png", img)

        count += num_events

