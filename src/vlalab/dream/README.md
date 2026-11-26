# vlalab/dream

One demo → thousands of randomised Newton / Steel worlds → policies. Nodes in
[stages/dream.py](../stages/dream.py); guides: [style dream](../../../docs/style_dream.md),
[UMI scene](../../../docs/style_dream_umi.md), [PPO from cameras](../../../docs/pixel_ppo.md).

| module | what it holds |
|---|---|
| `steel_env.py` | the toy worlds (cup, saucer, gripper on an x / y / z / yaw gantry), observation layout, success |
| `scene_env.py`, `scene_assets.py` | the worlds rebuilt from real2sim's reconstruction, geometry variants |
| `franka_hand.py` | newton-assets' Franka hand as the gripper |
| `planar_env.py` | the 2D worlds, and `make_env` for every backend |
| `shapes.py` | procedural cups, hulls, mass properties |
| `demo.py`, `style.py` | the demo, its style, retargeting, the style prior and reward |
| `world_model.py`, `agent.py` | the dreamer (RSSM) and the privileged policies (dreams + RL, PPO, dreams only) |
| `cameras.py` | the scene and wrist cameras in every world (Newton's `SensorCamera`) and visual randomisation |
| `pixel_ppo.py` | PPO from the cameras and the gripper's joints, curriculum from the privileged state |
| `render.py` | MP4 and USD renders of a few worlds |
