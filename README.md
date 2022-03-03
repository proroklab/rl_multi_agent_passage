# Multi-Agent RL passage formation for ICRA 2022
Repository containing the code base for training the multi-agent coordination policy used in the paper "[A Framework for Real-World Multi-Robot Systems Running Decentralized GNN-Based Policies](https://arxiv.org/abs/2111.01777)".

Supplementary video material:

[![Video preview](https://img.youtube.com/vi/COh-WLn4iO4/0.jpg)](https://www.youtube.com/watch?v=COh-WLn4iO4)

## Citation
If you use any part of this code in your research, please cite our paper:

```
@inproceedings{blumenkamp2022decentralizedgnn,
  title={A Framework for Real-World Multi-Robot Systems Running Decentralized GNN-Based Policies},
  author={Blumenkamp, Jan and Morad, Steven and Gielis, Jennifer and Li, Qingbiao and Prorok, Amanda},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2022},
  organization={IEEE}
}
```

## Setup
Clone with `git clone --recursive` to include submodules. Run `./build.sh` to set up a docker container for training. Training can be performed without docker by installing all requirements in the host system according to the `Dockerfile`.

## Train
Start training by running
```
./run.sh python3 src/train.py
```
The training will stop automatically after 5000 training iterations.

## Evaluate
Optionally, evaluate the policy performance by running
```
./run.sh python3 src/evaluate.py results/MultiPPO_simple_xxxx/checkpoint_yyyyyy
```
where the path to the model checkpoint has to be adapted accordingly.

## Export model
For ROS2 inference (refer [this](https://github.com/proroklab/ros2_multi_agent_passage) ROS2 project)

```
./run.sh python3 src/export.py results/MultiPPO_simple_xxxx/checkpoint_yyyyyy
```
The exported torchscript model will be saved in the same checkpoint folder and can then manually be copied to the correct location in the ROS2 project.
