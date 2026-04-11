import numpy as np

from evlib.dataloaders._mvsec_types import MVSECOdometryData


class TestMVSECOdometryData:
    def test_fields(self) -> None:
        odom = MVSECOdometryData(
            timestamps=np.array([0.0, 1.0], dtype=np.float64),
            linear_velocity=np.zeros((2, 3), dtype=np.float64),
            position=np.zeros((2, 3), dtype=np.float64),
            quaternion=np.zeros((2, 4), dtype=np.float64),
            angular_velocity=np.zeros((2, 3), dtype=np.float64),
        )
        assert odom.timestamps.shape == (2,)
        assert odom.linear_velocity.shape == (2, 3)

    def test_len(self) -> None:
        odom = MVSECOdometryData(
            timestamps=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            linear_velocity=np.zeros((3, 3), dtype=np.float64),
            position=np.zeros((3, 3), dtype=np.float64),
            quaternion=np.zeros((3, 4), dtype=np.float64),
            angular_velocity=np.zeros((3, 3), dtype=np.float64),
        )
        assert len(odom) == 3

    def test_frozen(self) -> None:
        odom = MVSECOdometryData(
            timestamps=np.array([0.0], dtype=np.float64),
            linear_velocity=np.zeros((1, 3), dtype=np.float64),
            position=np.zeros((1, 3), dtype=np.float64),
            quaternion=np.zeros((1, 4), dtype=np.float64),
            angular_velocity=np.zeros((1, 3), dtype=np.float64),
        )
        try:
            odom.timestamps = np.array([1.0])  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass
