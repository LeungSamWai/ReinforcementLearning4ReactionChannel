# Probing reaction channels via reinforcement learning
By A, B, C, D and E
## Introduction
Consolidating Hamiltonian terms into cliques allows 
simultaneous measurement and reduces shots, but prior knowledge of each clique, 
like amplitudes, is very limited. To tackle this challenge, we propose two novel shot assignment strategies based 
on the standard deviation estimation to refine the convergence of VQE and reduce the 
shot requirement. These strategies address measurement challenges in two distinct 
scenarios: when shots are overallocated or underallocated.

![image](quantum.png)

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
### Baseline strategies:
We first introduce two baseline strategies:
#### Uniform assignment:
```commandline
python qiskit-vqe-uniform_assignment.py --trial 1 --shots 240
```
#### Random assignment:
```commandline
python qiskit-vqe-random_assignment.py --trial 1 --shots 240
```
### Our strategies:
#### Variance-Minimized Shot Assignment:
```commandline
python qiskit-vqe-variance_minimized.py --trial 1 --shots 240 --std_shots 24
```
#### Variance-Preserved Shot Reduction:
```commandline
python qiskit-vqe-variance_preserved.py --trial 1 --shots 240 --std_shots 24
```

## Citing this paper
If you find this paper helps you in your research, please kindly cite:
