from collections import Counter

from crowdmotion.sim.mjlab.ablate_navigation import AblationCfg, arms


def test_factorial_has_both_crowds_seeds_and_exposure_matched_probes():
  cfg = AblationCfg()
  matrix = arms(cfg)
  assert len(matrix) == 48
  core = [r for r in matrix if r["group"] in {f"{mode}_g{g}" for mode in ("sparse", "potential", "all")
                                              for g in (75, 99)}]
  assert len(core) == 24
  assert set(Counter(r["group"] for r in core).values()) == {4}
  for row in matrix:
    assert row["iterations"] * row["navigation"]["rollout_s"] == cfg.iterations * 16
  for group in ("tracking_g99", "proximity_g99", "regularized_g99", "centered_g99", "lambda90", "rollout32",
                "no_time_bonus", "failure10"):
    assert {r["crowd"] for r in matrix if r["group"] == group} == {"reactive", "independent"}
