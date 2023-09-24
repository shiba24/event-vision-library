# Example using N-Caltech101 https://www.garrickorchard.com/datasets/n-caltech101
import numpy as np
from pathlib import Path
from evlib.codec import fileformat
from evlib.vis.view2d import events

# Setup iterator reader
bin_file_path = "./artifacts/sample_data/faces_easy_0001.bin"
ev_iter = fileformat.IteratorBinEvent(bin_file_path)
save_path = Path("./artifacts/results/faces_easy")
save_path.mkdir(parents=True, exist_ok=True)

# Iterate over the stream of events given a set temporal window
# TODO: This should probably be inside the event iterator
t_current = 0
t_window = 10e3  # 10 ms, as N-Caltech101 is in microseconds

for iter_data in ev_iter:
    t_max = iter_data['timestamp'][-1]
    i = 1
    while t_current < t_max:
        # Get t_window ms of the events stream and generate a plot
        mask = (iter_data['timestamp'] > t_current) & (iter_data['timestamp'] <= t_current + t_window)
        x_ = iter_data['x'][mask]
        y_ = iter_data['y'][mask]
        t_ = iter_data['t'][mask]
        p_ = iter_data['p'][mask]

        evs_np = np.vstack([y_, x_, t_, p_]).T
        image = events(evs_np, (ev_iter.size_y, ev_iter.size_x))
        # Show / Save image
        img_path = f'{str(save_path)}/{str(i).zfill(6)}.jpg'
        # image.show()
        image.save(img_path)

        t_current += t_window
        i += 1
