# Quantum-walk search in Motion (arXiv.)

This repository contains code related to the work presented in the paper titled 'Quantum Walk Search in Motion.' The code provides implementations, data, and tools used in the research project described in the paper.


## Paper Citation
If you use this code in your work, please cite our paper:

## Usage

The uses of code are straight forward, and only need to run the full python scripts for different algorithms discussed in article. The essential libraries such as numpy, scipy and matplotlib are used. 

## Description



* Essential_Function.py : This contain few useful function such as vector norm, complex conjugation etc. that will used throughout the notebook.
* QWSA.py : This contains code for quantum-walk based search algorithm done for open and closed two dimensional grid. 

* QWSA_Multipoint.py : We extend the algorithm with proposed mathod as give in paper for multiple point search algorithm that reveals the marked point as well as their order. 

![](Plots/single_layer_OBC_MAT_page-0001.jpg)

![](Plots/single_layer_PBC_MAT_page-0001.jpg)

Fig. (a) : Probability distribution at $t_\text{op}$ step for different layers with single marked point in each layer found using quantum-search algorithm single-layer amplification Top : Open boundary condition Bottom : Periodic boundary condition

![](Plots/multi_layer_OBC_MAT_page-0001.jpg)

![](Plots/multi_layer_PBC_MAT_page-0001.jpg)

Fig. (b) : Probability distribution at $t_\text{op}$ step for different layers with single marked point in each layer found using quantum-search algorithm multi-layer amplification Top : Open boundary condition Bottom : Periodic boundary condition



* Tracking.py : We use the formulation to propose the algorithm for particle tracking. The .gif or .mp4 file shows the results clearly showing the trajectory of particle as it pass through different layers.

![](Plots/movie.gif)

Fig. (c) : Probability distribution with time-steps in tracking problem


## Contact
For questions or further information, please contact Himanshu Sahu at [himanshusah1@iisc.ac.in].