"""Angular LiDAR ray pattern: directions and argument validation."""

import math

import pytest
import torch

from mjlab.sensor import LidarPatternCfg


def test_lidar_angles():
  offsets, directions = LidarPatternCfg(
    azimuth_samples=4, elevation_samples=1, elevation_range=(0, 0)
  ).generate_rays(None, "cpu")
  assert torch.count_nonzero(offsets) == 0
  torch.testing.assert_close(
    directions,
    torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [1.0, 0, 0], [0, 1.0, 0]]),
    atol=1e-6,
    rtol=0,
  )
  _, directions = LidarPatternCfg(
    azimuth_range=(0, 90),
    elevation_range=(-30, 30),
    azimuth_samples=2,
    elevation_samples=2,
  ).generate_rays(None, "cpu")
  torch.testing.assert_close(directions.norm(dim=-1), torch.ones(4))
  torch.testing.assert_close(directions[:, 2], torch.tensor([-0.5, -0.5, 0.5, 0.5]))


@pytest.mark.parametrize(
  "kwargs",
  [
    {"azimuth_samples": 0},
    {"elevation_samples": 0},
    {"azimuth_range": (0, 361)},
    {"elevation_range": (0, 91)},
    {"azimuth_range": (20, -20)},
    {"elevation_range": (0, math.nan)},
  ],
)
def test_invalid_lidar_pattern(kwargs):
  with pytest.raises(ValueError):
    LidarPatternCfg(**kwargs)
