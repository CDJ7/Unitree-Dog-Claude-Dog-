| Policy | Success 2-D | Success MuJoCo | Same outcome | Time, successful (2-D / MuJoCo, s) | Stain entries / ep (2-D / MuJoCo) | Steps with a foot in a stain / ep | Contact steps / ep (2-D / MuJoCo) |
|---|---|---|---|---|---|---|---|
| ppo | 0.517 | 0.667 | 0.85 | 16.2 / 24.9 | 0.62 / 1.53 | 30.6 | 52.7 / 47.6 |
| greedy | 0.217 | 0.183 | 0.97 | 6.7 / 5.8 | 0.50 / 0.77 | 145.5 | 14.2 / 44.4 |
| planner | 0.533 | 0.650 | 0.75 | 17.5 / 19.9 | 0.33 / 0.68 | 18.0 | 8.1 / 74.9 |

10 start poses x 6 test scenes = 60 episodes per policy and simulator; all evidence of the scene must be found; 180 s limit.
