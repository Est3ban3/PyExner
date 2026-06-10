# PyExner/integrators/registry.py
from typing import Callable, NamedTuple


from PyExner.solvers.registry import SolverBundle, SolverConfig
from PyExner.state.base import State

class SimState(NamedTuple):
    time: float
    out_freq: float
    end_time: float
    dt: float
    state: State
    cfl: float

class IntegratorConfig(NamedTuple):
    cfl: float
    end_time: float
    out_freq: float
    solver_config: SolverConfig
    solver_bundle: SolverBundle

class IntegratorBundle(NamedTuple):
    name: str
    config: Callable
    run_fn: Callable

INTEGRATOR_REGISTRY : dict[str, IntegratorBundle] = {}

def register_integrator_bundle(name: str):
    def decorator(bundle: IntegratorBundle):
        INTEGRATOR_REGISTRY[name] = bundle()
        return bundle()
    return decorator

def create_integrator_bundle(name: str):
    bundle = INTEGRATOR_REGISTRY[name]
    if bundle is None: 
        raise ValueError(f"Unknown integration scheme '{name}'. Available options: {list(SOLVER_REGISTRY.keys())}")
    return bundle
