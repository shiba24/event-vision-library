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

## Installation

You can install _Event Vision Library_ via [pip] from [PyPI]:

```console
$ pip install event-vision-library

# you can now `import evlib`
```

## Usage

Please see our [examples] and [documentation][read the docs].

## Features

- Python 3.7, 3.8, 3.9, 3.10
- Pure-python library
- Numpy and Torch compatibility.
- 🚧 This library is under construction and currently alpha version. The APIs may change significantly. Contributions and discussions are welcomed! 🚧

### Data

- [ ] Support different data types (.text, .raw, .hdf5, .npy, .aedat) for various file encoding of event data
- [ ] ROS bag files (optional, based on ROS installation)
- [ ] Support multiple existing dataset (e.g., ECD, MVSEC, DSEC, etc.)
- [ ] Support iterator-based loading and also block-based (random access) loading.

### Algorithms

- Have different off-the-shelf methods, ready to use:
  - [ ] Optical Flow estimation
  - [ ] Image reconstruction
  - [ ] Ego-motion estimation
  - more to come.
- [ ] C++ implementation and extension for faster execution (TODO)

### Log and Vsualization

- [ ] Various visualization for 2D/3D representation of events
- [ ] Useful logging

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Event Vision Library_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Acknowledgement

This project was generated from [Hypermodern Python Cookiecutter] template.

[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/shiba24/event-vision-library/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[examples]: https://github.com/shiba24/event-vision-library/blob/main/examples
[license]: https://github.com/shiba24/event-vision-library/blob/main/LICENSE
[contributor guide]: https://github.com/shiba24/event-vision-library/blob/main/CONTRIBUTING.md
[command-line reference]: https://event-vision-library.readthedocs.io/en/latest/usage.html
