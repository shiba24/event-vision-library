import numpy as np

def generate_events(
    n_events: int,
    height: int,
    width: int,
    tmin: float = 0.0,
    tmax: float = 0.5,
    dist: str = "uniform",
) -> np.ndarray:
    """Generate random events.

    Args:
        n_events (int) ... num of events
        height (int) ... height of the camera
        width (int) ... width of the camera
        tmin (float) ... timestamp min
        tmax (float) ... timestamp max
        dist (str) ... distribution of generated events. currently only "uniform" is supported.

    Returns:
        events (np.ndarray) ... [n_events x 4] numpy array.
            (y (height), x (width), t: [tmin, tmax], p: {0, 1})
    """
    if dist != "uniform":
        raise NotImplementedError
    y = np.random.randint(0, height, n_events)
    x = np.random.randint(0, width, n_events)
    t = np.random.uniform(tmin, tmax, n_events)
    t = np.sort(t)
    p = np.random.randint(0, 2, n_events)
    events: np.ndarray = np.concatenate([y[..., None], x[..., None], t[..., None], p[..., None]], axis=1)
    return events
