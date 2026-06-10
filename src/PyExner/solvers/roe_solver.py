# PyExner/solver/swe.py
import jax
import jax.numpy as jnp
from functools import partial
import mpi4jax
from mpi4py import MPI

# local imports
from PyExner.state.roe_state import RoeState

from PyExner.solvers.kernels.roe import compute_dt, roe_solve_2D, make_halo_exchange
from PyExner.solvers.registry import SolverConfig, SolverBundle, register_solver_bundle

def get_mask(state, *args, **kwargs):
    mask = jnp.zeros_like(state.h)
    mask = mask.at[1:-1, 1:-1].set(1)
    return mask

def config_fn_roe(state, mpi_handler, boundaries, dx, params):
    halo_exchange = make_halo_exchange(mpi_handler)

    return SolverConfig(
        mpi_handler = mpi_handler,
        boundaries = boundaries,
        dx = dx, 
        halo_exchange = halo_exchange
    )


def init_fn_roe(state: RoeState, mask, config: SolverConfig) -> RoeState:

    h = config.halo_exchange(state.h)
    hu = config.halo_exchange(state.hu)
    hv = config.halo_exchange(state.hv)
    z = config.halo_exchange(state.z)
    n = config.halo_exchange(state.n)

    return RoeState(h=h, hu=hu, hv=hv, z=z, n=n)

@partial(jax.jit, static_argnums=(4,))
def step_fn_roe(state: RoeState, time: float, dt: float, mask, config: SolverConfig) -> RoeState:    
    new_state = roe_solve_2D(state, dt, config.dx)

    # Update halos
    h = config.halo_exchange(new_state.h)
    hu = config.halo_exchange(new_state.hu)
    hv = config.halo_exchange(new_state.hv) 

    updated_state = RoeState(h=h, hu=hu, hv=hv, z=new_state.z, n=new_state.n)

    # Apply boundary conditions (must be pure)
    final_state = config.boundaries.apply(updated_state, time)

    return final_state

@partial(jax.jit, static_argnums=(3,))
def compute_dt_roe(state: RoeState, cfl: float, mask: jax.Array, config: SolverConfig) -> float:
    local_dt = cfl * compute_dt(state, mask, config.dx)
    global_dt = mpi4jax.allreduce(local_dt, op=MPI.MIN, comm=config.mpi_handler.cart_comm)
    return global_dt

@register_solver_bundle("Roe")
def solver_roe():
    return SolverBundle(
        name="Roe",
        config=config_fn_roe,
        mask_fn=get_mask,
        init_fn=init_fn_roe,
        step_fn=step_fn_roe,
        compute_dt_fn=compute_dt_roe
    )