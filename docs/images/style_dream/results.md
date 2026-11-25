**Newton + Steel (3D)** (2048 training worlds per policy, dreamer object error 4.8 mm at 1 step / 11.8 mm at 10 steps)

| method | held-out success (95% CI) | OOD success (95% CI) | style distance (mm) | tempo | success, heavy third |
|---|---|---|---|---|---|
| retarget + replay | 72.6% [70.6%, 74.4%] | 5.5% [4.6%, 6.5%] | 2.5 | 1.00 | 44% |
| RL only (PPO) | 95.8% [94.9%, 96.6%] | 44.5% [42.3%, 46.6%] | 3.0 | 0.97 | 91% |
| dreams only | 54.4% [52.3%, 56.6%] | 10.4% [9.2%, 11.8%] | 4.4 | 0.98 | 41% |
| dreams + RL (ours) | 96.3% [95.4%, 97.0%] | 49.1% [47.0%, 51.3%] | 3.7 | 1.00 | 92% |

**2D world** (1024 training worlds per policy, dreamer object error 1.5 mm at 1 step / 7.0 mm at 10 steps)

| method | held-out success (95% CI) | OOD success (95% CI) | style distance (mm) | tempo | success, heavy third |
|---|---|---|---|---|---|
| retarget + replay | 71.0% [69.6%, 72.3%] | 0.1% [0.0%, 0.3%] | 3.5 | 1.00 | 30% |
| RL only (PPO) | 95.3% [94.9%, 95.7%] ± 0.6% over 3 seeds | 21.5% [20.7%, 22.2%] | 3.6 | 0.99 | 86% |
| dreams only | 88.3% [87.7%, 88.8%] ± 2.0% over 3 seeds | 4.8% [4.4%, 5.2%] | 5.3 | 0.99 | 70% |
| dreams + RL (ours) | 93.5% [93.0%, 93.9%] ± 1.0% over 3 seeds | 12.8% [12.3%, 13.4%] | 3.4 | 0.99 | 81% |
