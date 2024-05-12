# RallySim_Evaluation
Evaluation code for RallySim : Simulated Environment with Continuous Control for Turn-based Multi-agent Reinforcement Learning. 
This repository is used to render simulated results of trained agents in RallySim. 
Currently this repo works only on Mac devices

## Requirements
ml-agents (https://github.com/Unity-Technologies/ml-agents/tree/release_20)
mlagents-envs (https://github.com/Unity-Technologies/ml-agents/tree/release_20)
torch 1.8.1

## Usage
Assuiming the repository is cloned into a local device and all required libiraries are installed.

### Low-level
Put a copy of the trained model to ./Badminton_Shrink-v4/models/ppo/ppo_actor_Badminton_Shrink-v4_Solid_seed[SEED]_iter[ITER]. Then go to /Badminton_Shrink-v4 and run
```
python3 evaluation.py --env=Badminton_Shrink-v4 --iter=ITER --seed=SEED
```

### High-level
Put a copy of the trained high-level model to ./Badminton_Shrink_Multi_HRL/models/ppo/ppo_actor_Badminton_Shrink_Multi_HRL_seed[SEED]_iter[ITER].pth. Then go to ./Badminton_Shrink_Multi_HRL and run
```
python3 Badminton_Shrink_Multi_HRL/evaluation.py --high_level_env=Badminton_Shrinc_Multi_HRL --iter_high=26 --seed_high=1 --low_level_env=Badminton_Shrink-v4 --seed=1 --iter=601 --train_high_level=1
```
