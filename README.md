# Event Vision Library

[![PyPI](https://img.shields.io/pypi/v/event-vision-library.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/event-vision-library.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/event-vision-library)][python version]
[![License](https://img.shields.io/pypi/l/event-vision-library)][license]

[![Read the documentation at https://event-vision-library.readthedocs.io/](https://img.shields.io/readthedocs/event-vision-library/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/shiba24/event-vision-library/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/shiba24/event-vision-library/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/event-vision-library/
[status]: https://pypi.org/project/event-vision-library/
[python version]: https://pypi.org/project/event-vision-library
[read the docs]: https://event-vision-library.readthedocs.io/
[tests]: https://github.com/shiba24/event-vision-library/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/shiba24/event-vision-library
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

### Algorithms

- Pure-python library
- Have different off-the-shelf methods, ready to use:
    - [ ] Optical Flow estimation
    - [ ] Image reconstruction
    - [ ] Ego-motion estimation
    - more to come.
- [ ] C++ implementation and extension for faster execution (TODO)

### Data

- [ ] Support different data types (.text, .raw, .hdf5, .npy, .aedat) for various file encoding of event data
- [ ] ROS bag files (optional, based on ROS installation)
- [ ] Support multiple existing dataset (e.g., ECD, MVSEC, DSEC, etc.)
- [ ] Support iterator-based loading and also block-based (random access) loading.

### Log and Vsualization

- [ ] Various visualization for 2D/3D representation of events
- [ ] Useful logging

## Requirements

- TODO

## Installation

You can install _Event Vision Library_ via [pip] from [PyPI]:

```console
$ pip install event-vision-library
```

## Usage

### Design note (data conversion)


```python
from evlib.codec import fileformat
output_hdf5 = 'sample_output.hdf5'

# Event data: from text file to hdf5
ev_iter = fileformat.IteratorTextEvent('./artifacts/sample_data/event.txt')
data_keys = {"x": "raw_events/x", "y": "raw_events/y", "t": "raw_events/t", "p": "raw_events/p"}
fileformat.convert_iterator_access_to_hdf5(ev_iter, output_hdf5, data_keys)

# Event data: from aedat to hdf5
ev_iter = fileformat.IteratorAedat4Event('./artifacts/sample_data/event.aedat')
data_keys = {"x": "raw_events/x", "y": "raw_events/y", "t": "raw_events/t", "p": "raw_events/p"}
fileformat.convert_iterator_access_to_hdf5(ev_iter, output_hdf5, data_keys)

# Frame data
fr_iter = fileformat.IteratorAedat4Frame('./artifacts/sample_data/event.aedat')
data_keys = {"frame": "frames/raw", "t": "frames/t"}
fileformat.convert_iterator_access_to_hdf5(fr_iter, output_hdf5, data_keys, image_keys=["frame"])

# from evk to hdf5
evk3_iter = fileformat.IteratorEvk3('./artifacts/sample_data/event.raw')
data_keys = {"x": "raw_events/x", "y": "raw_events/y", "t": "raw_events/t", "p": "raw_events/p"}
fileformat.convert_iterator_access_to_hdf5(evk3_iter, output_hdf5, data_keys)

# TODO Roabag
```

### Design note (data loader)

```python
from evlib import codec

# random (block) access
data_loader = codec.dataset.setup(fileformat=".txt", dataset_name="ecd", data_type="event")
events = data_loader.load_event(index1, index2) # n_events, 4

dense_flow = tasks.dense.optical_flow.cmax(events, **params) # returns 2, H, W
sparse_flow = tasks.sparse.optical_flow.triplet(events, **params) # returns n_events, 2
intensity = tasks.dense.intensity_reconstruction.linear_inverse(events, flow, **params)  # returns H, W
ang_vel = tasks.angular_velocity.cmax(events, **params)  # returns 3

# iterator access
data_loader = codec.dataset.setup(fileformat=".txt", dataset_name="ecd", data_type="event")
reconstructor = tasks.dense.intensity_reconstruction.e2vid(**param)
for events in data_loader:
    intensity = reconstructor.iterative_estimation(events, **params)  # returns H, W

# Accessing IMU  - low priority
data_loader = codec.dataset.setup(fileformat=".txt", dataset_name="ecd", data_type="imu")
for imu in data_loader:
    print(imu)

# Calibration - intrinsics
data_loader = codec.dataset.setup(fileformat=".txt", dataset_name="ecd", data_type="event")
reconstructor = tasks.dense.intensity_reconstruction.e2vid(**param)
calibrator = tasks.dense.calibration.e2vid_checkerboard()
for events in data_loader:
    intensity = reconstructor.iterative_run(events, **params)  # returns H, W
    calibrator.calculate_intrinsics(intensity)

# Calibration - extrinsics
data_loader1 = codec.dataset.setup(fileformat=".txt", dataset_name="ecd", data_type="event")
data_loader2 = codec.dataset.setup(fileformat=".raw", dataset_name="evk3", data_type="event")

reconstructor1 = tasks.dense.intensity_reconstruction.e2vid(**param)
reconstructor2 = tasks.dense.intensity_reconstruction.e2vid(**param)
calibrator = tasks.dense.calibration.e2vid_checkerboard()
# ... sync and load data as user specification

intensity1 = reconstructor1.iterative_run(events1, **params)  # returns H, W
intensity2 = reconstructor1.iterative_run(events2, **params)  # returns H, W
calibrator.calculate_homography(intensity1, intensity2)
```

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Event Vision Library_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/shiba24/event-vision-library/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/shiba24/event-vision-library/blob/main/LICENSE
[contributor guide]: https://github.com/shiba24/event-vision-library/blob/main/CONTRIBUTING.md
[command-line reference]: https://event-vision-library.readthedocs.io/en/latest/usage.html
