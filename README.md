# RIGEL - ReservoIr computinG based information sEcurity moduLe [![DOI](https://zenodo.org/badge/845505515.svg)](https://doi.org/10.5281/zenodo.15765839)

*<center> 混沌未分天地亂 茫茫渺渺無人見 </center>*
*<center> 自從盤古破鴻蒙 開闢從茲清濁辨 </center>*

## Table of Contents

- [I. Introduction](#i-introduction)
- [II. Environment Set-up](#ii-environment-set-up)
  * [i. Pre-requisites](#i-pre-requisites)
  * [ii. Managing python environment via anaconda](#ii-managing-python-environment-via-anaconda)
- [III. Development](#iii-development)
  * [i. For project maintainers](#i-for-project-maintainers)
  * [ii. For developers](#ii-for-developers)
- [IV. Citation](#iv-citation)
- [References](#references)

## I. Introduction

RIGEL is a python package designed for simulating and analyzing reservoir computing hardware systems, particularly focusing on chaotic systems. It provides tools for data acquisition, scientific calculus, dynamical system analysis, machine learning, deep learning, and visualization. 

The reservoir computing paradigm adopted in RIGEL is mainly based on Echo State Networks (ESNs) [^1][^2], a foundational model in reservoir computing. Reservoir computing itself can be regarded as a concise and randomized version of recurrent neural networks (RNNs), perserving the form of discrete-time dynamical systems mathematically. As such, it is particularly well-suited for tasks involving time series prediction, especially in chaotic systems where traditional methods may struggle. 

Beyond the core functionalities, RIGEL also includes modules for exploring the real-world applications of the simulated reservoir computing hardware systems, such as information security and more.

For more details, please refer to the [RIGEL documentation](https://majestyv.github.io/RIGEL/).

## II. Environment Set-up

### i. Pre-requisites

Recommended python version: 3.12 (which is used by myself). Anyway, in theory it should work with any python version >= 3.0.

Highly recommending using [anaconda](https://www.anaconda.com/) to manage your python environment. It is a free and open-source distribution of python and R programming languages for scientific computing, that aims to simplify package management and deployment.

Required packages:
- *Data acquisition*: [Pandas](https://pandas.pydata.org/)
- *Scientific calculus*: [Numpy](https://numpy.org/) & [Scipy](https://www.scipy.org/)
- *Dynamical system analysis*: [NoLiTSA](https://github.com/manu-mannattil/nolitsa) & [Numba](https://numba.pydata.org/)
- *Machine learning & Deep learning*: [Scikit-learn](https://scikit-learn.org/stable/) & [PyTorch](https://pytorch.org/)
- *Weight generation*: [Networkx](https://pypi.org/project/networkx/)
- *Visualization*: [OpenCV](https://opencv.org/), [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)
- *Visualizing progress bar*: [Tqdm](https://pypi.org/project/tqdm/)
- *Data encryption*: [PyCryptodome](https://pypi.org/project/pycryptodome/)

### ii. Managing python environment via anaconda

Using anaconda, we can easily manage our python environment and packages through the command line interface (CLI).

(1) Creating new anaconda environment
```Shell
$ conda create --name yourENV python=3.12
```
where `yourEnv` is the name of your environment.

(2) Activating the environment
```Shell
$ conda activate yourENV
```

(3) Installing packages

Part of the required packages can be quickly installed by calling the `requirements.txt` file, which is located in the 
root directory of this repository. The installation can be done via `conda` by the following command:
```Shell
$ conda install --yes --file requirements.txt
```
or `pip` by the following command:
```Shell
$ pip install -r requirements.txt
```
(***Reminding***: *When setting up virtual environments by anaconda, using `conda install` before `pip install` is usually beneficial for environment management.*)

## III. Development

For potential developer, feel free to clone this repository to your local computer and make improvements. If you are interested in this project, you can directly contact our group [Electronic Materials and Device @ CUHK](https://www.ee.cuhk.edu.hk/~ghhu/). We have a more powerful version on our organization github repository. The following is some use instructions for using [Git](https://git-scm.com/) to perform project development management.

Before starting, **always create your own branch from `main`**. By the way, here is a brief tutorial on basic git commands: [GitTest](https://github.com/MajestyV/GitTest).

### i. For project maintainers

```Shell
$ git clone <remote-repo-url>              # Clone the repository to your local machine

$ cd RIGEL                                # Change directory to the cloned repository

$ git remote add origin <remote-repo-url>  # Add the remote repository URL

$ git branch -M main                       # Rename the current branch to main

$ git push -uf origin main                 # Push the main branch to the remote repository
```

### ii. For developers

```Shell
$ git clone --branch <your-branch-name> <remote-repo-url>  # Clone the repository with your branch

$ git pull origin <your-branch-name>                       # Pull the latest changes from your branch

$ git add .                                                # Add all changes to the staging area
$ git commit -m "Whatever_you_did"                         # Commit your changes with a message
$ git push -u origin <your-branch-name>                    # Push your changes to your branch
```

## IV. Citation

If you find our work helpful, please refer to the following publication: [Analysis on reservoir activation with the nonlinearity harnessed from solution-processed molybdenum disulfide](https://arxiv.org/abs/2403.17676), *arXiv* (2024)

```
@article{RIGEL,
  Title={Analysis on reservoir activation with the nonlinearity harnessed from solution-processed molybdenum disulfide},
  Author={Songwei Liu, Yingyi Wen, Jingfang Pei, Yang Liu, Lekai Song, Pengyu Liu, Xiaoyue Fan, Wenchen Yang, Danmei Pan, Teng Ma, Yue Lin, Gang Wang, Guohua Hu},
  Eprint={arXiv:2403.17676},
  Year={2024}
}
```

## References
[^1]: [H. Jaeger. The "echo state" approach to analysing andtraining recurrent neural network, *GMD report* (2001)](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)

[^2]: [H. Jaeger, H. Haas. Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication, *Science* **304(5667)**, 78-80 (2004).](https://www.science.org/doi/10.1126/science.1091277)