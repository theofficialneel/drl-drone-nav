# Deep RL: Drone Navigation

Simple experiments on drone navigation using deep reinforcement algorithms. This repo is based on pybullets gym environment for drones ['gym-pybullet-drones'](https://github.com/utiasDSL/gym-pybullet-drones)


## Installation

This repo is built on python 3.10 on linux (Ubuntu 22.04)

Create a virtual env
```bash
$ python3 -m venv .venv
```

Must have
```bash
$ sudo apt install ffmpeg
```

Requirements
```bash
$ pip install -r requirements.txt
```

If there is an issue with gym-pybullet-drone gym env, visit [link](https://utiasdsl.github.io/gym-pybullet-drones/)

## Run

Run the python files in experiments/{exp}/*.py
```
$ python rl.py --args <arg_value>
```

View results using the view .py file
```
$ python rl_view.py --exp <location>
```

View tensorboard results of the training
```
$ tensorboard --logdir <location>
```

## Citation

```
@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      volume={},
      number={},
      pages={},
      doi={}
}
```
