# Probing reaction channels via reinforcement learning
By A, B, C, D and E

This repository provides implementation of the paper "Probing reaction channels via reinforcement learning" [arxiv](https://arxiv.org/pdf/2305.17531.pdf) on identifying the connective configuration with reinforcement learning (specially TD3). 

## Introduction

We propose a reinforcement learning based method to identify important configurations that connect reactant and product states along chemical reaction paths. Such important configurations are called connective configuration, characterized by the reactive probability to observe reactive trajectory crossing it. Utilizing these identified configuration, we can generate an ensemble of configurations that concentrate on the transition path ensemble, which can be further employed in a neural network-based partial differential equation solver to obtain an approximation solution of a restricted Backward Kolmogorov equation.

![image](RLconnetiveconf.png)

## Code structure
```commandline
Optimizing measurement
│   README.md    <-- You are here
│
│   qiskit-vqe-random_assignment.py  ---> Random assignment
│   qiskit-vqe-uniform_assignment.py ---> Uniform assignment
│   qiskit-vqe-variance_minimized.py ---> VMSA
│   qiskit-vqe-variance_preserved.py ---> VRSR
│   
│   readresult.ipynb ---> Result visualization
```
## How to run the code
There are three args: 
```commandline
--trial # the experiment number, which uses to name the result
--shots # Total number of shots for each iteration
--std_shots # Number of shots used to estimate the standard deviation for each clique
```
### Our strategies:
#### Variance-Minimized Shot Assignment:
```commandline
python qiskit-vqe-variance_minimized.py --trial 1 --shots 240 --std_shots 24
```

## Citing this paper
If you find this paper helps you in your research, please kindly cite:
```
@article{liang2023probing,
  title={Probing reaction channels via reinforcement learning},
  author={Liang, Senwei and Singh, Aditya N and Zhu, Yuanran and Limmer, David T and Yang, Chao},
  journal={arXiv preprint arXiv:2305.17531},
  year={2023}
}
```

## Acknowledgement
Many thanks to nikhilbarhate99 for his simple and clean framework of the [TD3 implementation](https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2).
