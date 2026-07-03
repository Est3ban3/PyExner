"""Transporte de 2do orden (MUSCL + HLL) frente al 1er orden.

El transporte HLL de 1er orden es demasiado difusivo: borra las plumas afiladas
de la EPB y subestima la tasa de crecimiento RT (en rt_instability_epb.py dio
gamma ~0.19x del teorico). Este script valida la reconstruccion MUSCL
(limitador MC, TVD) frente al esquema de 1er orden en cuatro frentes:

  A) ORDEN de convergencia en adveccion suave: 1er orden ~1, MUSCL ~2.
  B) Difusion en un frente AFILADO (deplecion tipo pluma): MUSCL conserva mucho
     mejor la variacion total / la profundidad del hueco.
  C) POSITIVIDAD: una deplecion del 99% no produce densidades negativas (la
     reconstruccion TVD acota los valores de cara entre los vecinos).
  D) RT en dominio doblemente periodico (setup de rt_instability_epb.py): MUSCL
     acerca el crecimiento al limite inercial y afila la pluma, comparado con
     1er orden.

Plasma frio (T_i=T_e=0) para A-C: el HLL degenera a upwind y la unica diferencia
con la solucion exacta es la difusion del esquema, que es justo lo que medimos.

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 python muscl_epb.py
"""

import math
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_twofluid import (
    EPBPhysParams,
    hll_flux,
    muscl_hll_flux_periodic,
    stack_state,
    unstack_state,
)
from PyExner.solvers.kernels.epb_sources import EPBSourceParams, implicit_source_solve
from PyExner.state.epb_twofluid_state import EPBTwoFluidState


PHYS_COLD = EPBPhysParams(Ti=0.0, Te=0.0)

L = 1.0
N0 = 1.0
V0 = 1.0
CFL = 0.4


# --------------------------------------------------------------------------- #
# Pasos periodicos 1er orden vs MUSCL (1D en x, plasma frio)                   #
# --------------------------------------------------------------------------- #

def _flux_first_order(Q, phys):
    """Flujo HLL de 1er orden en i+1/2 (reconstruccion constante)."""
    return hll_flux(Q, jnp.roll(Q, -1, axis=1), phys, "x")


def _l_op_x(Q, dx, phys, scheme):
    """Operador -div(F)/dx en x (periodico) para el esquema dado."""
    if scheme == "fo":
        Fp = _flux_first_order(Q, phys)
    else:
        Fp = muscl_hll_flux_periodic(Q, phys, "x")
    div = Fp - jnp.roll(Fp, 1, axis=1)     # F_{i+1/2} - F_{i-1/2}
    return -div / dx


def _step(Q, dt, dx, phys, scheme):
    if scheme == "fo":
        # 1er orden espacio + Euler explicito (estable, TVD).
        return Q + dt * _l_op_x(Q, dx, phys, "fo")
    # MUSCL: 2do orden espacio REQUIERE SSP-RK2 (Heun) en tiempo.
    L0 = _l_op_x(Q, dx, phys, "muscl")
    Q1 = Q + dt * L0
    L1 = _l_op_x(Q1, dx, phys, "muscl")
    return 0.5 * Q + 0.5 * (Q1 + dt * L1)


def _advect(Q0, dx, phys, scheme, t_final):
    step = jax.jit(lambda Q, dt: _step(Q, dt, dx, phys, scheme))
    dt = CFL * dx / V0
    Q = Q0
    t = 0.0
    while t < t_final - 1e-12:
        h = min(dt, t_final - t)
        Q = step(Q, h)
        t += h
    return Q


# --------------------------------------------------------------------------- #
# A) Orden de convergencia en adveccion suave                                  #
# --------------------------------------------------------------------------- #

def _gauss(x, shift, sigma=0.06):
    d = (x - L / 2 - shift + 0.5 * L) % L - 0.5 * L
    return N0 + 0.5 * np.exp(-d**2 / (2.0 * sigma**2))


def _build_smooth(nx):
    dx = L / nx
    xc = (np.arange(nx) + 0.5) * dx
    n = _gauss(xc, 0.0)
    e = PHYS_COLD.e
    j = e * n * V0
    z = np.zeros(nx)
    arr = np.stack([n, n, j, z, z, j, z, z], axis=-1)
    return jnp.asarray(arr[None, :, :]), xc, dx


def test_order():
    print("--- A) Orden de convergencia (adveccion suave) ---")
    t_final = 1.0          # un periodo completo (L=1, v0=1)
    grids = [64, 128, 256, 512]
    print("  scheme   nx     L1 error      orden")
    results = {}
    for scheme in ("fo", "muscl"):
        errs = []
        for nx in grids:
            Q0, xc, dx = _build_smooth(nx)
            Qf = _advect(Q0, dx, PHYS_COLD, scheme, t_final)
            n_num = np.asarray(Qf[0, :, 0])
            n_exact = _gauss(xc, V0 * t_final)   # vuelve al inicio
            err = float(np.mean(np.abs(n_num - n_exact)))
            errs.append(err)
        orders = [np.log(errs[i] / errs[i + 1]) / np.log(2.0) for i in range(len(errs) - 1)]
        results[scheme] = (errs, orders)
        name = "1er ord" if scheme == "fo" else "MUSCL  "
        for i, nx in enumerate(grids):
            o = f"{orders[i-1]:5.2f}" if i > 0 else "  -  "
            print(f"  {name} {nx:4d}   {errs[i]:.4e}    {o}")
    mean_fo = np.mean(results["fo"][1])
    mean_mu = np.mean(results["muscl"][1])
    print(f"  orden medio: 1er={mean_fo:.2f}  MUSCL={mean_mu:.2f}")
    ok = (mean_mu > 1.5) and (mean_mu > mean_fo + 0.4)
    print("  [OK] MUSCL converge a orden ~2 y supera al de 1er orden" if ok
          else "  [FAIL] MUSCL no mejora el orden")
    return ok


# --------------------------------------------------------------------------- #
# B) Difusion en un frente afilado (pluma)                                     #
# --------------------------------------------------------------------------- #

def _build_plume(nx):
    """Deplecion tipo "top-hat" suavizado: hueco de densidad del 80%."""
    dx = L / nx
    xc = (np.arange(nx) + 0.5) * dx
    d = (xc - L / 2 + 0.5 * L) % L - 0.5 * L
    hole = 0.8 * (np.abs(d) < 0.15).astype(float)
    n = N0 - hole
    e = PHYS_COLD.e
    j = e * n * V0
    z = np.zeros(nx)
    arr = np.stack([n, n, j, z, z, j, z, z], axis=-1)
    return jnp.asarray(arr[None, :, :]), xc, dx


def test_sharp_front():
    print("\n--- B) Frente afilado (pluma de deplecion) ---")
    nx = 256
    t_final = 1.0     # un periodo: la solucion exacta vuelve al inicio
    Q0, xc, dx = _build_plume(nx)
    n0 = np.asarray(Q0[0, :, 0])

    out = {}
    for scheme in ("fo", "muscl"):
        Qf = _advect(Q0, dx, PHYS_COLD, scheme, t_final)
        n = np.asarray(Qf[0, :, 0])
        # Error L1 frente a la solucion exacta (= inicial tras un periodo).
        l1 = float(np.mean(np.abs(n - n0)))
        # Ancho de transicion: nº de celdas con 0.1 < (N0-n)/0.8 < 0.9 (rampa).
        frac = (N0 - n) / 0.8
        width = int(np.sum((frac > 0.1) & (frac < 0.9)))
        out[scheme] = (l1, width, n)
        name = "1er ord" if scheme == "fo" else "MUSCL  "
        print(f"  {name}: error L1={l1:.4e}  celdas en rampa (difusion)={width}")

    l1_fo, w_fo, _ = out["fo"]
    l1_mu, w_mu, _ = out["muscl"]
    # MUSCL debe tener MENOR error L1 y frentes mas estrechos (menos celdas de rampa).
    ok = (l1_mu < 0.6 * l1_fo) and (w_mu < w_fo)
    print(f"  L1 MUSCL/1er = {l1_mu / l1_fo:.2f}x   "
          f"ancho de rampa MUSCL/1er = {w_mu}/{w_fo}")
    print("  [OK] MUSCL conserva el frente afilado mucho mejor" if ok
          else "  [FAIL] MUSCL no mejora la nitidez del frente")
    return ok


# --------------------------------------------------------------------------- #
# C) Positividad con deplecion extrema (99%)                                   #
# --------------------------------------------------------------------------- #

def test_positivity():
    print("\n--- C) Positividad (deplecion del 99%) ---")
    nx = 256
    dx = L / nx
    xc = (np.arange(nx) + 0.5) * dx
    d = (xc - L / 2 + 0.5 * L) % L - 0.5 * L
    n = N0 - 0.99 * np.exp(-d**2 / (2.0 * 0.03**2))   # casi vacio en el centro
    e = PHYS_COLD.e
    j = e * n * V0
    z = np.zeros(nx)
    Q0 = jnp.asarray(np.stack([n, n, j, z, z, j, z, z], axis=-1)[None, :, :])

    nmin_global = float(n.min())
    step = jax.jit(lambda Q, dt: _step(Q, dt, dx, PHYS_COLD, "muscl"))
    dt = CFL * dx / V0
    Q = Q0
    for _ in range(400):
        Q = step(Q, dt)
        nmin_global = min(nmin_global, float(jnp.min(Q[0, :, 0])))
    print(f"  min densidad global (400 pasos MUSCL) = {nmin_global:.3e}  "
          f"(inicial {float(n.min()):.3e})")
    ok = nmin_global >= 0.0
    print("  [OK] TVD preserva positividad sin floor" if ok
          else "  [FAIL] aparecieron densidades negativas")
    return ok


# --------------------------------------------------------------------------- #
# D) RT doblemente periodico: MUSCL vs 1er orden (setup de rt_instability)     #
# --------------------------------------------------------------------------- #

# Parametros identicos a rt_instability_epb.py (plasma frio, doblemente periodico).
RT_PHYS = EPBPhysParams(Ti=0.0, Te=0.0)
RT_LX = 4.0
RT_NX = 64
RT_DX = RT_LX / RT_NX
RT_G = 1.0
RT_B = 1.0
RT_NLOW = 0.2
RT_NHIGH = 1.0
RT_ZC = 2.0
RT_W = 0.3
RT_EPS = 1e-2
RT_K = 2
RT_CFL = 0.3
RT_NSTEPS = 300


def _rt_build():
    xc = (np.arange(RT_NX) + 0.5) * RT_DX
    zc = (np.arange(RT_NX) + 0.5) * RT_DX
    X, Z = np.meshgrid(xc, zc, indexing="xy")
    base = RT_NLOW + (RT_NHIGH - RT_NLOW) * np.exp(-((Z - RT_ZC) ** 2) / (2.0 * RT_W**2))
    env = np.exp(-((Z - (RT_ZC - RT_W)) ** 2) / (2.0 * RT_W**2))
    n = base * (1.0 + RT_EPS * np.cos(2.0 * np.pi * RT_K * X / RT_LX) * env)
    n_i = jnp.asarray(n)
    z = jnp.zeros((RT_NX, RT_NX))
    return EPBTwoFluidState(n_i=n_i, n_e=n_i, j_ix=z, j_iy=z, j_iz=z, j_ex=z, j_ey=z, j_ez=z)


def _rt_transport(Q, dt, scheme):
    if scheme == "fo":
        Fx = hll_flux(Q, jnp.roll(Q, -1, axis=1), RT_PHYS, "x")
        Gz = hll_flux(Q, jnp.roll(Q, -1, axis=0), RT_PHYS, "z")
        divx = Fx - jnp.roll(Fx, 1, axis=1)
        divz = Gz - jnp.roll(Gz, 1, axis=0)
        return Q - (dt / RT_DX) * (divx + divz)

    # MUSCL: SSP-RK2 (estable y 2do orden).
    def L(Qx):
        Fx = muscl_hll_flux_periodic(Qx, RT_PHYS, "x")
        Gz = muscl_hll_flux_periodic(Qx, RT_PHYS, "z")
        divx = Fx - jnp.roll(Fx, 1, axis=1)
        divz = Gz - jnp.roll(Gz, 1, axis=0)
        return -(divx + divz) / RT_DX

    L0 = L(Q)
    Q1 = Q + dt * L0
    L1 = L(Q1)
    return 0.5 * Q + 0.5 * (Q1 + dt * L1)


def _rt_amp(state):
    n = np.asarray(state.n_i)
    return float(np.sum((n - n.mean(axis=1, keepdims=True)) ** 2))


def _rt_run(scheme):
    src = EPBSourceParams(gz=-RT_G, By=RT_B)
    state = _rt_build()
    v_char = max(RT_PHYS.Mi, RT_PHYS.Me) * RT_G / (RT_PHYS.e * RT_B)
    dt = RT_CFL * RT_DX / max(v_char, 1e-3)

    def imex(s):
        Q = _rt_transport(stack_state(s), dt, scheme)
        s2 = unstack_state(Q)
        return implicit_source_solve(s2, dt, RT_PHYS, src, RT_DX)

    step = jax.jit(imex)
    amps = [_rt_amp(state)]
    nmin = float(np.asarray(state.n_i).min())
    for _ in range(RT_NSTEPS):
        state = step(state)
        amps.append(_rt_amp(state))
        nmin = min(nmin, float(np.asarray(state.n_i).min()))
    return np.array(amps), nmin, dt


def _growth_rate(amps, dt):
    n = len(amps)
    lo, hi = n // 8, n // 2
    t = np.arange(lo, hi) * dt
    y = 0.5 * np.log(np.maximum(amps[lo:hi], 1e-300))
    A = np.vstack([t, np.ones_like(t)]).T
    slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return slope


def test_rt_growth():
    print("\n--- D) RT doblemente periodico: MUSCL vs 1er orden ---")
    gamma_theory = math.sqrt(RT_G / RT_W)
    amps_fo, nmin_fo, dt = _rt_run("fo")
    amps_mu, nmin_mu, _ = _rt_run("muscl")
    g_fo = _growth_rate(amps_fo, dt)
    g_mu = _growth_rate(amps_mu, dt)
    print(f"  gamma teorico (inercial) sqrt(g/Ln) = {gamma_theory:.3f}")
    print(f"  1er orden: gamma={g_fo:.3f}  ratio={g_fo/gamma_theory:.2f}  "
          f"factor A={amps_fo[-1]/amps_fo[0]:.1f}  min n={nmin_fo:.4f}")
    print(f"  MUSCL    : gamma={g_mu:.3f}  ratio={g_mu/gamma_theory:.2f}  "
          f"factor A={amps_mu[-1]/amps_mu[0]:.1f}  min n={nmin_mu:.4f}")
    ok = True
    # MUSCL debe dar un crecimiento MAYOR (menos difusion) y seguir positivo.
    if not (g_mu > g_fo):
        print("  [FAIL] MUSCL no aumenta la tasa de crecimiento"); ok = False
    else:
        print(f"  [OK] MUSCL aumenta gamma {g_mu/g_fo:.2f}x (menos difusion numerica)")
    if nmin_mu <= 0.0:
        print("  [FAIL] densidad no positiva con MUSCL"); ok = False
    else:
        print("  [OK] positividad mantenida con MUSCL en la RT")
    return ok


def main():
    print("=== Transporte de 2do orden (MUSCL + HLL) ===\n")
    results = [
        test_order(),
        test_sharp_front(),
        test_positivity(),
        test_rt_growth(),
    ]
    ok = all(results)
    print("\n" + ("MUSCL: TODAS LAS PRUEBAS PASARON" if ok else "MUSCL: HAY FALLOS"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
