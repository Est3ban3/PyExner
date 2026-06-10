# PyExner/solver/registry.py

import jax

from PyExner.parallel.mpi_utils import Parallel
from PyExner.domain.boundary_registry import BoundaryManager

from typing import Callable, NamedTuple

class SolverConfig(NamedTuple):
    mpi_handler: Parallel
    boundaries: BoundaryManager
    dx: float
    halo_exchange: Callable[[jax.Array], jax.Array]
    compute_G: Callable = None
    compute_n: Callable = None
    # Scheme-specific physical parameters (e.g. EPB_TwoFluid: charges, masses,
    # temperatures). Kept optional to preserve backward compatibility with the
    # hydraulic branches (Roe, Roe Exner), which leave it as None.
    phys: object = None
    # Scheme-specific source/background parameters (e.g. EPB_TwoFluid: gravity,
    # E_0, B, neutral wind, collision frequencies). Optional, defaults to None.
    src: object = None

class SolverBundle(NamedTuple):
    name: str
    config: Callable
    mask_fn: Callable
    init_fn: Callable
    step_fn: Callable
    compute_dt_fn: Callable
    # Optional implicit stiff-source operator (used by IMEX integrators). The
    # hydraulic branches (Roe, Roe Exner) have no stiff sources and leave it
    # as None; the explicit integrators never call it.
    source_fn: Callable = None

SOLVER_REGISTRY: dict[str, SolverBundle] = {}

def register_solver_bundle(name: str):
    def decorator(bundle: Callable):
        SOLVER_REGISTRY[name] = bundle()
        return bundle()
    return decorator

def create_solver_bundle(name: str):
    bundle = SOLVER_REGISTRY[name]
    if bundle is None:
        raise ValueError(f"Unknown solver scheme '{name}'. Available options: {list(SOLVER_REGISTRY.keys())}")
    return bundle