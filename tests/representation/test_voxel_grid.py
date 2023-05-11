from evlib.utils import basics as basic_utils
from evlib.representation import VoxelGrid


def test_build_voxel_grid_shape():    # type: ignore
    ne = 500
    nbins = 5
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)
    voxel_grid_builder = VoxelGrid((height, width), num_bins=nbins)
    voxel_grid = voxel_grid_builder(ev)
    assert voxel_grid.shape == (nbins, height, width)