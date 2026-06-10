"""Solver bundle de la rama ``EPB_TwoFluid`` (modelo 2.5D de dos fluidos).

ESTADO DE IMPLEMENTACION (Fase 2 — esqueleto / wiring):

Este bundle registra el camino funcional ``flux_scheme: EPB_TwoFluid`` dentro
del framework (estado + solver + integrador + I/O), pero el nucleo numerico es
todavia un placeholder:

    * ``step_fn``        : intercambia halos y aplica contornos, pero el paso de
                           transporte (``transport_step``) es identidad. No hay
                           flujos hiperbolicos ni fuentes todavia.
    * ``compute_dt_fn``  : usa el dt de referencia placeholder del kernel.

Las piezas fisicas se implementan en fases posteriores:
    * Fase 3: flujos hiperbolicos F(Q), G(Q) + velocidades de onda + CFL real.
    * Fase 4: contornos transmisivos e I/O de los 8 campos.
    * Fase 5: operador de fuentes rigidas S(Q) + solve eliptico de phi.
    * Fase 6: integrador IMEX.

La separacion entre transporte, fuentes y solve electrostatico se mantiene
estricta desde el esqueleto.
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

    Devuelve ``stack([blocked, b_mask])`` para ser compatible con el contrato
    del integrador (que usa ``mask[0]`` para la escritura de salida). La
    semantica fina de la mascara se finaliza en la Fase 3 junto con el solver
    hiperbolico.
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
    # Paso 1: transporte hiperbolico HLL (slice puro — Fase 3).
    state = transport_step(state, dt, config.dx, mask, config.phys)

    # Paso 2: sincronizar halos de las 8 componentes conservadas.
    state = halo_exchange_all(state, config.halo_exchange)

    # Paso 3: aplicar condiciones de contorno (debe ser puro).
    state = config.boundaries.apply(state, time)

    return state


def compute_dt_epb(state: EPBTwoFluidState, cfl: float, mask: jax.Array, config: SolverConfig) -> float:
    local_dt = cfl * compute_dt_2D(state, config.dx, mask, config.phys)
    global_dt = mpi4jax.allreduce(local_dt, op=MPI.MIN, comm=config.mpi_handler.cart_comm)
    return global_dt


@partial(jax.jit, static_argnums=(4,))
def source_fn_epb(state: EPBTwoFluidState, time: float, dt: float, mask, config: SolverConfig) -> EPBTwoFluidState:
    """Parte implicita del IMEX: solve rigido de fuentes + halo + contornos.

    Resuelve localmente (I - dt A) j = j + dt b (colisiones + magnetica
    implicitas, campo electrico lagged via solve eliptico). Mantiene la
    separacion: el transporte ya se aplico en ``step_fn``; aqui solo actuan las
    fuentes y se re-sincroniza el halo / contorno.
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
