# Unitree-Dog-Claude-Dog-
Store the files needed to run the Unitree Go2

## Test 2 (ENG 6010): Go2 crime-scene search dog

A simulated Unitree Go2 walks through a MuJoCo crime scene, records a multi-sensor dataset (Q2), recognises
which zone of the scene it is in from its camera (Q1, ResNet18), and learns an evidence-search policy with PPO
that is deployed on the full MuJoCo Go2 (Q3). Write-up, results and reproduction commands:
[Test2/README.md](Test2/README.md).


## Phase 2: crime-scene investigator, now on the real dog

[Phase_2_Investigator/](Phase_2_Investigator/README.md):
- **Simulation (Stage A):** the Go2 searches simulated crime scenes with bloodstain physics.
- **Real dog (Stage R, in progress):** the Go2 explores the lab on its own, builds a LiDAR map, tags each
  part with a zone (lab_1, lab_2, open_area, shelves) and walks to the zone you ask for. Terminal-only
  walkthrough for the Linux laptop: [Phase_2_Investigator/real_robot/README.md](Phase_2_Investigator/real_robot/README.md).
