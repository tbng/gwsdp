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

## Python RiNNAL solver

```python
from gwsdp import solve_gw_sdp, solve_fused_gw_sdp, solve_rinnal

pi, P, value = solve_gw_sdp(C1, C2, p, q, solver="rinnal")
pi, P, value = solve_fused_gw_sdp(M, C1, C2, p, q, alpha=0.8, solver="rinnal")
pi, P, value, info = solve_rinnal(C1, C2, p, q, tol=1e-5,
                                  max_iters=300, return_info=True)
```

`pi` is the transport plan, `P` is the lifted second moment, and `value`
is the squared-loss objective evaluated on the returned lift. The index of
entry `pi[i, j]` in `P` is `i + len(p) * j`. For a higher-rank lift the
GW cost of `pi` can exceed `value`.

The solver specializes Hou, Tang and Toh's low-rank augmented Lagrangian
method to the continuous transport constraints. It enforces both sets of
lifted marginal equalities in factor coordinates and uses SciPy L-BFGS for
the inner minimizations. Small problems use the lifted projected-gradient
rank correction from RiNNAL+. Larger problems use a negative-curvature
direction and exploit the Kronecker structure of the GW cost. The multiplier
and returned moment still require memory proportional to `(len(p)*len(q))**2`.
On large, uniform square problems, an assignment and local swaps supply
a feasible initial plan. For symmetric structure matrices, conditional
distance-profile assignments also initialize the multipliers and lower bound.
See the [RiNNAL paper](https://arxiv.org/abs/2502.13849),
[RiNNAL+ paper](https://arxiv.org/abs/2507.13776), and authors' MATLAB
[RiNNAL](https://github.com/HouDiOpt/RiNNAL) and
[RiNNAL+](https://github.com/HouDiOpt/RiNNALplus) releases.

`max_iters` counts outer iterations. At the limit, the solver emits a
`RuntimeWarning` and returns the current iterate. With `return_info=True`,
inspect `converged`, `primal_residual` (Frobenius norm of the negative
entries of the homogenized lift), and `relative_gap` (distance from the
iterate objective to the numerical lower bound, divided by one plus the
absolute objective). The separate `lower_bound` and `upper_bound` give
a bracket from dual information and a feasible coupling. For costs with
large magnitude, compare absolute objective errors in the original units.

## Run the validation experiments

Run these commands from the repository root with its dependencies and
`pytest` installed.

```bash
python -m pytest -q
python -m benchmarks.benchmark_rinnal --case gaussian --size 6 6 --seed 0
python -m benchmarks.benchmark_rinnal --case gaussian --size 6 6 --seed 0 --solver scs --tol 1e-8 --max-iters 100000
python -m benchmarks.benchmark_rinnal --case sbm-fgw --size 6 6 --seed 5
python -m benchmarks.benchmark_rinnal --case gaussian --size 50 50 --seed 0 --max-iters 300
```

The benchmark seeds both random streams and times the solver after generating
the inputs. Each invocation prints a JSON record with objective values,
constraint residuals, solver status, wall time, and process peak memory. Run
each size and backend in its own process for comparable memory measurements.

## Contacts ##

If you have any problems, please open an issue in this repository or ping an email to [tuanbinhs@gmail.com](mailto:tuanbinhs@gmail.com).
