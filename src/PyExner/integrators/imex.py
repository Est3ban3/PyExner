# PyExner/integrators/imex.py
"""Integrador IMEX (transporte explicito + fuentes rigidas implicitas).

FASE 6 — Introduce un integrador de tipo IMEX como rama adicional del framework,
reutilizando el MISMO contrato de ejecucion (configuracion, bucle temporal,
escritura de salida) que ``Forward Euler``.

Razon numerica: el paso hiperbolico esta gobernado por velocidades de onda
(CFL convectivo), mientras que las fuentes locales (colisiones, girofrecuencia
magnetica) pueden ser mucho mas rapidas. Integrarlas de forma explicita
obligaria a un dt diminuto o seria directamente inestable. El IMEX resuelve:

    1. Transporte EXPLICITO (``step_fn`` del solver: flujos HLL + halo + BC).
    2. Fuentes IMPLICITAS (``source_fn`` del solver: algebra local por celda
       (I - dt A) j = j + dt b, con E lagged del solve eliptico).

Es un splitting de Lie de primer orden, consistente con el dt convectivo del
``compute_dt_fn``. Si el ``SolverBundle`` no define ``source_fn`` (ramas
hidraulicas), el IMEX se reduce exactamente a Forward Euler.
"""

from PyExner.utils.constants import TIMESTEP_TOL
from PyExner.integrators.registry import (
    IntegratorConfig,
    IntegratorBundle,
    register_integrator_bundle,
)
from PyExner.state.base import State
from typing import NamedTuple

import jax
import jax.numpy as jnp

import time as timer


def config_fn_imex(cfl, end_time, out_freq, solver_bundle, solver_config):
    return IntegratorConfig(
        cfl=cfl,
        end_time=end_time,
        out_freq=out_freq,
        solver_config=solver_config,
        solver_bundle=solver_bundle,
    )


class SimState(NamedTuple):
    time: float
    out_freq: float
    end_time: float
    dt: float
    state: State
    cfl: float


def cond_fn(simstate):
    next_out_time = ((simstate.time / simstate.out_freq).astype(jnp.int32) + 1) * simstate.out_freq
    next_target = jnp.minimum(next_out_time, simstate.end_time)
    return simstate.time + simstate.dt < next_target - TIMESTEP_TOL


def _imex_step(solver_bundle, state, time, dt, mask, solver_config):
    """Un paso IMEX: transporte explicito seguido de fuentes implicitas."""
    # Parte explicita (Ex): transporte hiperbolico + halo + contornos.
    state = solver_bundle.step_fn(state, time, dt, mask, solver_config)
    # Parte implicita (Im): fuentes rigidas locales (si el solver las define).
    if solver_bundle.source_fn is not None:
        state = solver_bundle.source_fn(state, time, dt, mask, solver_config)
    return state


def make_body_fn(solver_bundle, mask, solver_config):
    def body_fn(simstate):
        new_dt = solver_bundle.compute_dt_fn(simstate.state, simstate.cfl, mask, solver_config)
        new_state = _imex_step(solver_bundle, simstate.state, simstate.time, new_dt, mask, solver_config)
        return SimState(
            time=simstate.time + new_dt,
            out_freq=simstate.out_freq,
            end_time=simstate.end_time,
            dt=new_dt,
            state=new_state,
            cfl=simstate.cfl,
        )
    return body_fn


def run_fn_imex(state: State, config: IntegratorConfig, io, mesh, b_mask) -> State:
    iters = 0
    time = 0.0
    numOut = 0
    dt = 0.0

    mask = config.solver_bundle.mask_fn(state, config.solver_config.mpi_handler, b_mask)
    body_fn = make_body_fn(config.solver_bundle, mask, config.solver_config)
    state = config.solver_bundle.init_fn(state, mask, config.solver_config)

    rank = config.solver_config.mpi_handler.rank

    if io is not None:
        io.write_state(state, mesh, mask[0])

    a = timer.perf_counter()

    while time < config.end_time - TIMESTEP_TOL:

        # === ADVANCE UNTIL NEXT OUTPUT WINDOW ===
        simstate = SimState(
            time=time,
            out_freq=config.out_freq,
            end_time=config.end_time,
            dt=dt,
            state=state,
            cfl=config.cfl,
        )

        simstate = jax.lax.while_loop(cond_fn, body_fn, simstate)

        time = simstate.time
        state = simstate.state

        # === NEXT STEP SIZE ===
        dt = config.solver_bundle.compute_dt_fn(state, config.cfl, mask, config.solver_config)

        next_out_time = (int(time / config.out_freq) + 1) * config.out_freq
        next_target = min(next_out_time, config.end_time)

        hit_output = time + dt >= next_target - TIMESTEP_TOL

        if hit_output:
            dt_probe = next_target - time

            # === PURE PROBE (NO COMMIT) ===
            state_io = _imex_step(
                config.solver_bundle, state, time, dt_probe, mask, config.solver_config
            )

            if io is not None:
                io.write_state(state_io, mesh, mask[0])

            if rank == 0:
                print(f"[IMEX] Iteration: {iters}  Time: {next_target:.6f}")
                print(f"[IMEX] Probe dt: {dt_probe:.9f}")
                print(f"[IO Writer] File: {numOut} written. Time: {next_target:.6f}")

            numOut += 1

        # === ALWAYS DO PHYSICS STEP WITH ORIGINAL CFL dt ===
        state = _imex_step(
            config.solver_bundle, state, time, dt, mask, config.solver_config
        )

        time += dt
        iters += 1

    b = timer.perf_counter()

    if rank == 0:
        import hashlib
        now = timer.time()

        key = f"{dt:.10f}_{now:.6f}"
        h = hashlib.sha256(key.encode()).hexdigest()[:12]
        filename = f"timing_{h}.txt"

        with open(filename, "w") as f:
            print(f"Time: {b-a}", file=f)

    return state


@register_integrator_bundle("IMEX")
def integrator_imex():
    return IntegratorBundle(
        name="IMEX",
        config=config_fn_imex,
        run_fn=run_fn_imex,
    )
