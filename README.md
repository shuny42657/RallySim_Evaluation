# RallySim_Evaluation
Evaluation code for RallySim : Simulated Environment with Continuous Control for Turn-based Multi-agent Reinforcement Learning. 
This repository is used to render simulated results of trained agents in RallySim. 
Currently this repo works only on Mac devices

## Requirements
ml-agents (https://github.com/Unity-Technologies/ml-agents/tree/release_20).
mlagents-envs (https://github.com/Unity-Technologies/ml-agents/tree/release_20).
torch 1.8.1.

## Usage
Assuiming the repository is cloned into a local device and all required libiraries are installed.

### Low-level
Put a copy of the trained model to ./Low_level/models/ppo/ppo_actor_Low_level_seed[SEED]_iter[ITER].pth. Then go to ./Low_level and run
```
python3 evaluation.py --env=Low_level --iter=ITER --seed=SEED
```
If you just want to run the model which in the /models folder by default, simply run
```
python3 evaluation.py
```
By calling an argument "--test-num" you can adjust the number of episode you render (its default value is set to 10).

### High-level
Put a copy of the trained high-level model to ./High_level/models/ppo/ppo_actor_High_level_seed[SEED_LOW]_iter[ITER_LOW].pth. Then go to ./High_level and run
```
python3 evaluation.py --iter_high=ITER_HIGH --seed_high=SEED_HIGH --seed_low=SEED_LOW --iter_low=ITER_LOW
```

Or use the default models by simply running
```
python3 evaluation.py
```
By calling an argument "--test-num" you can adjust the number of episode you render (its default value is set to 5).
