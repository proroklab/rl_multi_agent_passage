# Multi-Agent RL object tracking for CoRL 2020

## Setup
Clone with `git clone --recursive` to include submodules! Run `./build.sh` to set up a docker container for training.

## Train
```
./run.sh python3 src/train.py
```

## Evaluate
```
./run.sh python3 src/evaluate.py results/MultiPPO_simple_xxxx/checkpoint_yyyyyy
```

## Export model
For ROS2 inference (refer [this](https://github.com/proroklab/ros2_multi_agent_passage) ROS2 project)

```
./run.sh python3 src/export.py results/MultiPPO_simple_xxxx/checkpoint_yyyyyy
```