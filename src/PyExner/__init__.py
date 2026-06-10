
from importlib import import_module


_EXPORT_MAP = {
    "Mesh2D": ("PyExner.domain.mesh", "Mesh2D"),
    "run_driver": ("PyExner.runtime.driver", "run_driver"),
    "RoeState": ("PyExner.state", "RoeState"),
    "STATE_REGISTRY": ("PyExner.state", "STATE_REGISTRY"),
    "create_state": ("PyExner.state", "create_state"),
    "register_state": ("PyExner.state", "register_state"),
    "SOLVER_REGISTRY": ("PyExner.solvers", "SOLVER_REGISTRY"),
    "create_solver_bundle": ("PyExner.solvers", "create_solver_bundle"),
    "register_solver_bundle": ("PyExner.solvers", "register_solver_bundle"),
    "INTEGRATOR_REGISTRY": ("PyExner.integrators", "INTEGRATOR_REGISTRY"),
    "create_integrator_bundle": ("PyExner.integrators", "create_integrator_bundle"),
    "register_integrator_bundle": ("PyExner.integrators", "register_integrator_bundle"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module 'PyExner' has no attribute '{name}'") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "Mesh2D",
    "run_driver",
    "RoeState",
    "STATE_REGISTRY",
    "create_state",
    "register_state",
    "SOLVER_REGISTRY",
    "create_solver_bundle",
    "register_solver_bundle",
    "INTEGRATOR_REGISTRY",
    "create_integrator_bundle",
    "register_integrator_bundle",
]
