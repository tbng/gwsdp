# [NeuRIPS 2024] Semidefinite Relaxations of the Gromov-Wasserstein Distance

<div  align="center">

<a  href="https://blog.nus.edu.sg/chenjunyu/"  target="_blank">Junyu&nbsp;Chen</a> &emsp; <b>&middot;</b> &emsp;
<a  href="https://tbng.github.io/"  target="_blank">Binh T.&nbsp;Nguyen</a> &emsp; <b>&middot;</b> &emsp;
<a  href="https://openreview.net/profile?id=~Shang_Hui_Koh1"  target="_blank">Shang Hui&nbsp;Koh</a> &emsp; <b>&middot;</b> &emsp;
<a  href="https://yssoh.github.io/"  target="_blank">Yong Sheng&nbsp;Soh</a>
<br>
<a  href="https://arxiv.org/abs/2312.14572">[Paper]</a> &emsp;&emsp; 
<a  href="https://openreview.net/forum?id=rM3FFH1mqk">[OpenReview]</a> &emsp;&emsp;

</div>

<br>

The Gromov-Wasserstein (GW) distance is an extension of the optimal transport problem that allows one to match objects between incomparable spaces.  At its core, the GW distance is specified as the solution of a non-convex quadratic program and is not known to be tractable to solve.  In particular, existing solvers for the GW distance are only able to find locally optimal solutions.  In this work, we propose a semi-definite programming (SDP) relaxation of the GW distance. The relaxation can be viewed as the Lagrangian dual of the GW distance augmented with constraints that relate to the linear and quadratic terms of transportation plans. In particular, our relaxation provides a tractable (polynomial-time) algorithm to compute globally optimal transportation plans (in some instances) together with an accompanying proof of global optimality.  Our numerical experiments suggest that the proposed relaxation is strong in that it frequently computes the globally optimal solution. 

**Please CITE** our paper whenever utilizing this repository to generate published results or incorporating it into other software.

```
@inproceedings{
	chen2024semidefinite,
	title={Semidefinite Relaxations of the Gromov-Wasserstein Distance},
	author={Chen, Junyu and Nguyen, Binh and Koh, Shang Hui and Soh, Yong Sheng},
	booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
	year={2024},
	url={https://openreview.net/forum?id=rM3FFH1mqk}
}
```

## Installation

It is recommended to create a `conda` env and install the required packages with our config file by doing the followings
```
conda create -n gwsdp python=3.8
pip install -r requirements.txt
pip install -e .
```

To use Mosek solver, you also need a valid license file `mosek.lic`.

## How to run experiments
WIP

## Contacts ##

If you have any problems, please open an issue in this repository or ping an email to [tuanbinhs@gmail.com](mailto:tuanbinhs@gmail.com).
