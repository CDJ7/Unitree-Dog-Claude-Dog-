| Policy | Success 2-D | Success MuJoCo | Same outcome | Time, successful (2-D / MuJoCo, s) | Stain entries / ep (2-D / MuJoCo) | Steps with a foot in a stain / ep | Contact steps / ep (2-D / MuJoCo) |
|---|---|---|---|---|---|---|---|
| ppo | 0.533 | 0.633 | 0.80 | 36.9 / 24.0 | 2.00 / 1.68 | 3.0 | 22.4 / 65.5 |
| greedy | 0.250 | 0.183 | 0.93 | 9.7 / 8.7 | 0.70 / 0.85 | 148.1 | 0.0 / 53.1 |
| planner | 0.733 | 0.700 | 0.93 | 20.5 / 18.9 | 0.60 / 0.78 | 4.0 | 0.8 / 95.7 |

10 start poses x 6 test scenes = 60 episodes per policy and simulator; all evidence of the scene must be found; 180 s limit.
