| Policy | Success 2-D | Success MuJoCo | Same outcome | Time, successful (2-D / MuJoCo, s) | Stain entries / ep (2-D / MuJoCo) | Steps with a foot in a stain / ep | Contact steps / ep (2-D / MuJoCo) |
|---|---|---|---|---|---|---|---|
| ppo | 0.200 | 0.467 | 0.73 | 14.3 / 30.7 | 0.57 / 1.40 | 26.5 | 14.0 / 42.5 |
| greedy | 0.217 | 0.167 | 0.95 | 6.7 / 5.9 | 0.50 / 0.80 | 155.8 | 14.2 / 24.5 |
| planner | 0.533 | 0.667 | 0.77 | 17.5 / 21.8 | 0.33 / 0.82 | 3.9 | 8.1 / 65.2 |

10 start poses x 6 test scenes = 60 episodes per policy and simulator; all evidence of the scene must be found; 180 s limit.
