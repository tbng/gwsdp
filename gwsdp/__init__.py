"""
GWSDP: Gromov-Wasserstein Semidefinite Programming Package

This package provides tools for solving optimal transport problems using Gromov-Wasserstein distances
through semidefinite programming (SDP) approaches. It includes solvers for standard Gromov-Wasserstein
problems and fused Gromov-Wasserstein problems that combine structural and feature similarities.

Key modules:
- solvers: Contains SDP-based algorithms for Gromov-Wasserstein and fused GW problems
- utils: Utility functions for cost tensor computation and result evaluation
- generate_dataset: Tools for generating synthetic datasets for testing and benchmarking

For more information about Gromov-Wasserstein distances and their applications in optimal transport,
refer to the accompanying documentation.
"""

# Version
__version__ = "0.1.0"

# Public API
from gwsdp.solvers import solve_gw_sdp, solve_fused_gw_sdp
from gwsdp.utils import (
    cost_tensor_numpy, cost_tensor_pytorch,
    gw_from_cost, gw_from_cost_pytorch,
    fgw_from_cost, optimality_gap, approximation_gap,
    sinkhorn_scaling
)
from gwsdp.generate_dataset import generate_sample, generate_sample_two
