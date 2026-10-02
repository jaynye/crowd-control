"""Corridor fixture: geometric oracles for human/room scans, contacts and world isolation.

Runs on CPU (or CUDA when present; FORCE_CPU=1 forces CPU). Skips without mjlab.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mjlab")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.scene import Scene  # noqa: E402
from mjlab.sensor import LidarPatternCfg, RayCastSensorCfg  # noqa: E402

from crowdmotion.sim.mjlab.corridor.humans import PrimitiveHumanCfg  # noqa: E402
from crowdmotion.sim.mjlab.corridor.scene import make_crowd_env_cfg  # noqa: E402
from crowdmotion.sim.mjlab.corridor.walking import CorridorWalk, CorridorWalkCfg  # noqa: E402


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def test_maximum_crowd_starts_without_overlap():
  cfg = make_crowd_env_cfg(num_humans=8)
  model = Scene(cfg.scene, device="cpu").compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 0


@pytest.fixture(scope="module")
def crowd_env():
  cfg = make_crowd_env_cfg(num_envs=2, num_humans=1, moving=False)
  cfg.scene.entities["human_0"] = PrimitiveHumanCfg().entity_cfg()
  cfg.scene.entities["scanner"].init_state.pos = (-5, 0, 1.22)
  sensor = cfg.scene.sensors[0]
  assert isinstance(sensor, RayCastSensorCfg)
  sensor.pattern = LidarPatternCfg(
    azimuth_range=(0, 0),
    elevation_range=(0, 0),
    azimuth_samples=1,
    elevation_samples=1,
  )
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  yield env
  env.close()


def test_human_hit_then_wall_after_motion(crowd_env):
  env = crowd_env
  env.reset()
  sensor = env.scene["lidar"]
  torso_id = env.sim.mj_model.geom("human_0/torso").id
  wall_id = env.sim.mj_model.geom("room/wall_front").id
  torch.testing.assert_close(
    sensor.data.distances,
    torch.full((2, 1), 4.86, device=env.device),
    atol=1e-4,
    rtol=0,
  )
  assert torch.all(sensor.data.geom_ids == torso_id)
  # Move only world 0. World 1 must retain its own human geometry.
  env.scene["human_0"].write_joint_state_to_sim(
    torch.tensor([[0.0, 1.0, 0.0]], device=env.device),
    torch.zeros(1, 3, device=env.device),
    env_ids=torch.tensor([0], device=env.device),
  )
  env.sim.forward()
  env.sim.sense()
  assert sensor.data.geom_ids[:, 0].tolist() == [wall_id, torso_id]
  torch.testing.assert_close(
    sensor.data.distances[:, 0],
    torch.tensor([11.0, 4.86], device=env.device),
    atol=1e-4,
    rtol=0,
  )
  env.reset(env_ids=torch.tensor([0], device=env.device))
  assert torch.all(sensor.data.geom_ids == torso_id)


def test_near_occluder_does_not_reveal_wall(crowd_env):
  env = crowd_env
  env.reset()
  sensor = env.scene["lidar"]
  # Torso front at x=-0.14; put the scanner 5 cm in front of it.
  pose = torch.tensor([[-0.19, 0, 1.22, 1, 0, 0, 0]], device=env.device).repeat(2, 1)
  env.scene["scanner"].write_mocap_pose_to_sim(
    pose, env_ids=torch.arange(env.num_envs, device=env.device)
  )
  env.sim.forward()
  env.sim.sense()
  assert torch.all(sensor.data.distances == -1)
  assert torch.all(sensor.data.geom_ids == -1)
  assert torch.count_nonzero(sensor.data.normals_w) == 0
  torch.testing.assert_close(sensor.data.hit_pos_w[:, 0], pose[:, :3])


def test_wall_occludes_human(crowd_env):
  env = crowd_env
  env.reset()
  # Place the human behind the corridor's front wall.
  env.scene["human_0"].write_joint_state_to_sim(
    torch.tensor([[7.0, 0, 0]], device=env.device).repeat(2, 1),
    torch.zeros(2, 3, device=env.device),
  )
  env.sim.forward()
  env.sim.sense()
  assert torch.all(
    env.scene["lidar"].data.geom_ids == env.sim.mj_model.geom("room/wall_front").id
  )


def test_human_wall_physical_contact(crowd_env):
  env = crowd_env
  env.reset()
  human = env.scene["human_0"]
  assert env.sim.mj_model.body("human_0/body").mass[0] == pytest.approx(70)
  human.write_joint_state_to_sim(
    torch.tensor([[5.93, 0, 0]], device=env.device).repeat(2, 1),
    torch.zeros(2, 3, device=env.device),
  )
  env.sim.forward()
  for _ in range(4):
    env.sim.step()
    env.scene.update(env.physics_dt)
  contact = env.scene["human_0_contacts"].data
  assert torch.all(contact.found > 0)
  assert torch.all(contact.force_history.norm(dim=-1).amax(dim=-1) > 0)
  assert torch.isfinite(human.data.joint_vel).all()
  assert torch.all(human.data.joint_vel[:, 0] < 0)


def test_primitive_scan_matches_cpu_rays():
  cfg = make_crowd_env_cfg(num_humans=1, moving=False)
  cfg.scene.entities["human_0"] = PrimitiveHumanCfg().entity_cfg()
  cfg.scene.entities["scanner"].init_state.pos = (-5, 0, 1.22)
  sensor_cfg = cfg.scene.sensors[0]
  assert isinstance(sensor_cfg, RayCastSensorCfg)
  sensor_cfg.pattern = LidarPatternCfg(
    azimuth_range=(-5, 5),
    elevation_range=(-15, 10),
    azimuth_samples=21,
    elevation_samples=26,
  )
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    env.reset()
    model = env.sim.mj_model
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    cpu.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
    cpu.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
    mujoco.mj_forward(model, cpu)
    _, directions = sensor_cfg.pattern.generate_rays(model, "cpu")
    geom_id = np.zeros(1, dtype=np.int32)
    groups = np.array([1, 1, 1, 0, 1, 0], dtype=np.uint8)
    expected_ids, expected_distances = [], []
    for direction in directions.numpy():
      distance = mujoco.mj_ray(
        model,
        cpu,
        np.array([-5.0, 0, 1.22]),
        direction.astype(np.float64),
        groups,
        True,
        model.body("scanner/scanner").id,
        geom_id,
      )
      expected_ids.append(geom_id[0])
      expected_distances.append(distance)
    actual = env.scene["lidar"].data
    np.testing.assert_allclose(actual.distances[0].cpu(), expected_distances, atol=2e-4)
    np.testing.assert_array_equal(actual.geom_ids[0].cpu(), expected_ids)
    hit_names = {model.geom(int(i)).name for i in expected_ids}
    assert {
      f"human_0/{part}"
      for part in (
        "head",
        "torso",
        "pelvis",
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
      )
    } <= hit_names
  finally:
    env.close()


def test_floor_ceiling_and_wall_distances():
  cfg = make_crowd_env_cfg(num_humans=0)
  cfg.scene.entities["scanner"].init_state.pos = (0, 0, 1)
  sensor = cfg.scene.sensors[0]
  assert isinstance(sensor, RayCastSensorCfg)
  sensor.pattern = LidarPatternCfg(
    azimuth_range=(0, 0),
    elevation_range=(-90, 90),
    azimuth_samples=1,
    elevation_samples=3,
  )
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    env.reset()
    actual = env.scene["lidar"].data
    torch.testing.assert_close(
      actual.distances[0],
      torch.tensor([1.0, 6.0, 1.8], device=env.device),
      atol=1e-5,
      rtol=0,
    )
    assert [env.sim.mj_model.geom(i).name for i in actual.geom_ids[0].tolist()] == [
      "room/floor",
      "room/wall_front",
      "room/ceiling",
    ]
  finally:
    env.close()


def test_walking_and_partial_reset():
  cfg = make_crowd_env_cfg(
    num_envs=2, num_humans=2, seed=7, azimuth_samples=36, elevation_samples=4
  )
  for command in cfg.commands.values():
    assert isinstance(command, CorridorWalkCfg)
    command.pause_probability = 0.0
    command.resampling_time_range = (1.0, 1.5)
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    env.reset()
    human = env.scene["human_0"]
    initial = human.data.joint_pos.clone()
    first_scan = env.scene["lidar"].data.distances.clone()
    action = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(100):
      env.step(action)
    assert torch.all((human.data.joint_pos[:, :2] - initial[:, :2]).norm(dim=-1) > 0.1)
    assert not torch.allclose(first_scan, env.scene["lidar"].data.distances)
    assert torch.isfinite(human.data.joint_pos).all()
    assert torch.all(human.data.joint_pos[:, :2].abs() < 5.6)
    for name in cfg.commands:
      term = env.command_manager.get_term(name)
      assert isinstance(term, CorridorWalk)
      assert torch.all(term.command_counter >= 2)
    command = env.command_manager.get_term("human_0")
    assert isinstance(command, CorridorWalk)
    # Resetting one world must not change another world's pose, scan, or intent.
    before_pose = human.data.joint_pos[1].clone()
    before_scan = env.scene["lidar"].data.distances[1].clone()
    before_command = command.command[1].clone()
    before_timer = command.time_left[1].clone()
    before_rng = command._rng[1].clone()
    env.reset(env_ids=torch.tensor([0], device=env.device))
    torch.testing.assert_close(human.data.joint_pos[1], before_pose)
    torch.testing.assert_close(env.scene["lidar"].data.distances[1], before_scan)
    torch.testing.assert_close(command.command[1], before_command)
    torch.testing.assert_close(command.time_left[1], before_timer)
    torch.testing.assert_close(command._rng[1], before_rng)
    torch.testing.assert_close(human.data.joint_pos[0], initial[0])
  finally:
    env.close()
