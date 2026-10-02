"""Primitive-human corridor fixture for LiDAR and contact development (no robot policy).

A 12 x 4 x 2.8 m room, rigid primitive pedestrians on planar x/y/yaw joints with
finite-force velocity actuators (`humans.PrimitiveHumanCfg`), seeded waypoint
walking (`walking.CorridorWalk`) and a fixed angular LiDAR rig
(`scene.make_crowd_env_cfg`). `demo` runs it in a viewer or headless and plots
saved scans. Moved here from the mjlab clone, which keeps only the general
sensor features (LiDAR pattern, minimum range, hit geom ids).
"""
