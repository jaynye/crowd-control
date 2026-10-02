# TANGO (humanoid navigation via a whole-body VLA)

Li, Chen, Li, Cao, Ren, Tomizuka, Shah. "TANGO: Humanoid Navigation in Cluttered
Environments with a Whole-Body Vision-Language-Action Model", arXiv:2609.09158.
Raised by the user on 2026-09-30 as an alternative to RL
([learning paradigm](../decisions/learning-paradigm.md)).

**Source caveat:** the facts below came from a summarizing web fetch of the arXiv
abstract and HTML pages on 2026-09-30, not a full reading. Check the paper before
relying on a specific number.

## What it does

- **Task:** G1 navigation through **static** clutter: narrow lateral passages,
  ground-level obstacles to step over, overhead obstacles. No pedestrians or dynamic
  obstacles.
- **Observations:** egocentric RGB, front and downward cameras, at 2 Hz. The paper lists
  RGB-only input as a limitation and says depth or LiDAR should help.
- **Expert data, Plan-Edit-Track (PET):**
  1. **Plan:** A* with an obstacle-aware cost, plus heading adjustment for narrow
     passages.
  2. **Edit:** whole-body IK with four objectives: reference posture, foot contact and
     landing targets, a center-of-mass target, and potential-field link forces applied
     as SoftMimic-style pseudo-forces. Gait adaptation moves landings past obstacles and
     raises swing-foot clearance. Reference gaits come from SONIC driven by velocity
     commands.
  3. **Track:** SONIC tracks the edited motion in simulation as a feasibility and
     collision filter; failures are discarded.
- **Scale:** 64,633 trajectories over 578 source scenes (205 VLNVerse, 373 SAGE-3D).
  PET took 86 and rendering 125 RTX PRO 6000 GPU-hours (211 total). IsaacSim renders the
  data; MuJoCo runs low-level tracking.
- **Policy:** Qwen2.5-VL-7B (system 2) plus a flow-matching DiT action expert
  (system 1) producing 29-DoF targets for the SONIC tracker (system 0). Loss
  `L_CE + 20·L_FM`, one epoch at lr 1e-5, real-time action chunking at training time.
  Trained on 896 A100 GPU-hours. **No RL stage.**
- **Result:** zero-shot transfer to a real G1 with no real navigation data. Stated
  limits: the tracker (for example, stairs) and RGB-only perception.

## Relevance to crowd-control

- The **robot** side of TANGO (PET) handles static scenes only. Crowds would need a
  planner that accounts for the humans' future motion. Baked humans make that motion
  known exactly to a privileged expert.
- The **human** side is absent from TANGO. Its pipeline implies the crowd is generated
  offline per episode, which suits ARDY or Kimodo directly, without a graph runtime.
- See the [learning paradigm](../decisions/learning-paradigm.md) decision for what each
  branch needs.
