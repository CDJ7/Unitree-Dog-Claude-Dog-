| Policy | Success | Evidence found | Time, successful eps (s) | Contamination entries / ep | Eps with contamination | Collision steps / ep | Severe-collision ends | Path (m) |
|---|---|---|---|---|---|---|---|---|
| random | 0.017 | 0.136 | 90.0 | 1.30 | 0.22 | 31.2 | 0.00 | 21.0 |
| greedy | 0.367 | 0.553 | 20.1 | 1.25 | 0.47 | 2.9 | 0.03 | 4.6 |
| ppo | 0.117 | 0.302 | 45.9 | 0.50 | 0.27 | 18.5 | 0.45 | 6.9 |

60 episodes per policy on the test layouts ['s2', 'r1854', 'r3101', 'r3102', 'r3103', 'r3104']; max episode length 180 s.
