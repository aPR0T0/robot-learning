**Steel, rebuilt UMI scene (mouse to pad)** (2048 training worlds per policy, dreamer object error 7.0 mm at 1 step / 20.3 mm at 10 steps)

| method | held-out success (95% CI) | OOD success (95% CI) | style distance (mm) | tempo | success, heavy third |
|---|---|---|---|---|---|
| retarget + replay | 46.0% [43.9%, 48.2%] | 0.5% [0.3%, 0.9%] | 3.8 | 1.00 | 25% |
| RL only (PPO) | 97.6% [96.9%, 98.2%] | 52.3% [50.1%, 54.5%] | 4.9 | 1.02 | 97% |
| dreams only | 0.2% [0.1%, 0.6%] | 0.0% [0.0%, 0.2%] | 10.8 | 0.98 | 0% |
| dreams + RL (ours) | 95.8% [94.8%, 96.5%] | 39.6% [37.5%, 41.7%] | 5.9 | 1.02 | 93% |
