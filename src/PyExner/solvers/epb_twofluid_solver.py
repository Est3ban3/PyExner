# PyExner/solvers/epb_twofluid_solver.py
"""Solver bundle de la rama EPB_TwoFluid.

Conecta el modelo de dos fluidos al framework con el mismo contrato que las
ramas hidraulicas (config / mask / init / step / dt), mas un ``source_fn``
que el integrador IMEX invoca despues del transporte:

    step_fn   : transporte HLL explicito + halos + contornos
    source_fn : fuentes rigidas implicitas (colisiones, B, E) + halos + contornos

El transporte de produccion es el de 1er orden (el MUSCL periodico existe en
el kernel pero necesita ghost cells para contornos fisicos, asi que todavia
no esta cableado aca).
"""

import jax
import jax.numpy as jnp
import mpi4jax
from mpi4py import MPI

from functools import partial

from PyExner.state.epb_twofluid_state import EPBTwoFluidState

from PyExner.solvers.kernels.epb_twofluid import (
    compute_dt_2D,
    transport_step,
    make_halo_exchange,
    halo_exchange_all,
    EPBPhysParams,
)
from PyExner.solvers.kernels.epb_sources import EPBSourceParams, implicit_source_solve
from PyExner.solvers.registry import SolverConfig, SolverBundle, register_solver_bundle


def get_mask(state, mpi_handler, b_mask):
    """Mascara de celdas bloqueadas + mascara de contorno.

    Devuelve stack([blocked, b_mask]): el integrador usa mask[0] para la
    escritura de salida. Se bloquea el anillo exterior del dominio global
    (solo en los rangos que tocan el borde), salvo donde hay contorno.
    """
    blocked = jnp.zeros_like(state.n_i, dtype=bool)

    coords = mpi_handler.coords
    dims = mpi_handler.dims

    if coords[0] == 0:
        blocked = blocked.at[0, :].set(True)
    if coords[0] == dims[0] - 1:
        blocked = blocked.at[-1, :].set(True)
    if coords[1] == 0:
        blocked = blocked.at[:, 0].set(True)
    if coords[1] == dims[1] - 1:
        blocked = blocked.at[:, -1].set(True)

    blocked = jnp.where(b_mask, False, blocked)

    return jnp.stack([blocked, b_mask], axis=0)


def config_fn_epb(state, mpi_handler, boundaries, dx, params):
    halo_exchange = make_halo_exchange(mpi_handler)
    phys = EPBPhysParams.from_params(params)
    src = EPBSourceParams.from_params(params)

    return SolverConfig(
        mpi_handler=mpi_handler,
        boundaries=boundaries,
        dx=dx,
        halo_exchange=halo_exchange,
        phys=phys,
        src=src,
    )


def init_fn_epb(state: EPBTwoFluidState, mask, config: SolverConfig) -> EPBTwoFluidState:
    state = halo_exchange_all(state, config.halo_exchange)
    state = config.boundaries.apply(state, 0.0)
    return state


@partial(jax.jit, static_argnums=(4,))
def step_fn_epb(state: EPBTwoFluidState, time: float, dt: float, mask, config: SolverConfig) -> EPBTwoFluidState:
    # Transporte hiperbolico explicito.
    state = transport_step(state, dt, config.dx, mask, config.phys)

    # Halos de las 8 componentes y contornos (deben ser puros para el jit).
    state = halo_exchange_all(state, config.halo_exchange)
    state = config.boundaries.apply(state, time)

    return state


def compute_dt_epb(state: EPBTwoFluidState, cfl: float, mask: jax.Array, config: SolverConfig) -> float:
    local_dt = cfl * compute_dt_2D(state, config.dx, mask, config.phys)
    global_dt = mpi4jax.allreduce(local_dt, op=MPI.MIN, comm=config.mpi_handler.cart_comm)
    return global_dt


@partial(jax.jit, static_argnums=(4,))
def source_fn_epb(state: EPBTwoFluidState, time: float, dt: float, mask, config: SolverConfig) -> EPBTwoFluidState:
    """Parte implicita del IMEX.

    Resuelve por celda (I - dt A) j = j + dt b (colisiones y giro implicitos,
    E congelado via el solve eliptico) y re-sincroniza halos y contornos. El
    transporte ya paso por step_fn; aca solo actuan las fuentes.
    """
    state = implicit_source_solve(state, dt, config.phys, config.src, config.dx)
    state = halo_exchange_all(state, config.halo_exchange)
    state = config.boundaries.apply(state, time)
    return state


@register_solver_bundle("EPB_TwoFluid")
def solver_epb_twofluid():
    return SolverBundle(
        name="EPB_TwoFluid",
        config=config_fn_epb,
        mask_fn=get_mask,
        init_fn=init_fn_epb,
        step_fn=step_fn_epb,
        compute_dt_fn=compute_dt_epb,
        source_fn=source_fn_epb,
    )
