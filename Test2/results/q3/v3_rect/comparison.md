| Policy | Success | Evidence found | Time, successful eps (s) | Contamination entries / ep | Eps with contamination | Collision steps / ep | Severe-collision ends | Path (m) |
|---|---|---|---|---|---|---|---|---|
| random | 0.050 | 0.163 | 145.7 | 1.05 | 0.25 | 40.0 | 0.00 | 20.9 |
| greedy | 0.300 | 0.475 | 13.9 | 0.58 | 0.45 | 28.6 | 0.62 | 2.7 |
| planner | 0.700 | 0.786 | 18.5 | 0.40 | 0.33 | 24.7 | 0.15 | 3.0 |
| ppo | 0.250 | 0.481 | 14.8 | 0.90 | 0.28 | 13.4 | 0.23 | 13.8 |

60 episodes per policy on the test layouts ['s2', 'r1854', 'r3101', 'r3102', 'r3103', 'r3104']; max episode length 180 s.
