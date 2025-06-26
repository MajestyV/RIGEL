# RIGEL - ReservoIr computinG based information sEcurity moduLe

*<center> 混沌未分天地亂 茫茫渺渺無人見 </center>*
*<center> 自從盤古破鴻蒙 開闢從茲清濁辨 </center>*

## I. Introduction

RIGEL is a python package designed for simulating and analyzing reservoir computing hardware systems, particularly focusing on chaotic systems. It provides tools for data acquisition, scientific calculus, dynamical system analysis, machine learning, deep learning, and visualization. 

The reservoir computing paradigm adopted in RIGEL is mainly based on Echo State Networks (ESNs) [^1][^2], a foundational model in reservoir computing. Reservoir computing itself can be regarded as a concise and randomized version of recurrent neural networks (RNNs), perserving the form of discrete-time dynamical systems mathematically. As such, it is particularly well-suited for tasks involving time series prediction, especially in chaotic systems where traditional methods may struggle. 

Beyond the core functionalities, RIGEL also includes modules for exploring the real-world applications of the simulated reservoir computing hardware systems, such as information security and more.

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
conda create --name yourENV python=3.12
```
where `yourEnv` is the name of your environment.

(2) Activating the environment
```Shell
conda activate yourENV
```

(3) Installing packages

Part of the required packages can be quickly installed by calling the `requirements.txt` file, which is located in the 
root directory of this repository. The installation can be done via `conda` by the following command:
```Shell
conda install --yes --file requirements.txt
```
or `pip` by the following command:
```Shell
pip install -r requirements.txt
```
(***Reminding***: *When setting up virtual environments by anaconda, using `conda install` before `pip install` is usually beneficial for environment management.*)

## III. Citation

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
[^1] [H. Jaeger. The "echo state" approach to analysing andtraining recurrent neural network, *GMD report* (2001)](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)

[^2] [H. Jaeger, H. Haas. Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication, *Science* 304:5667, 78-80 (2004).](https://www.science.org/doi/10.1126/science.1091277)