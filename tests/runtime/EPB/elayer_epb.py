"""Capa E como carga de conductancia (shunt) sobre el solve electrostatico.

Valida el solve electrostatico de COEFICIENTE VARIABLE

    D^-( sigma_f G^+ phi ) = D^- J,        sigma = Sigma_total / Sigma_ref,

y el cierre fisico del cortocircuito de capa E: las capas E conjugadas drenan
la carga de polarizacion de la region F y diluyen E_p por el factor de
apantallamiento F_s = Sigma_F/(Sigma_F + Sigma_E). Es el mecanismo que explica
por que la EPB es un fenomeno POST-PUESTA-DE-SOL (de dia Sigma_E apantalla).

Bloques:

  A) EQUIVALENCIA: sigma = 1 reproduce ``solve_phi`` (~maquina).
  B) SHUNT ANALITICO: sigma uniforme = (1+R) => phi = phi_1/(1+R) EXACTO
     (la unica solucion del sistema lineal escalado). Precision de maquina.
  C) LIMPIEZA CON COEFICIENTE VARIABLE: para sigma(x,z) no uniforme el campo
     corregido cumple D^-J - D^-(sigma_f G^+ phi) = residual CG ~ maquina
     (la garantia de poisson_epb.py se hereda con coeficiente variable).
  D) APANTALLAMIENTO DE LA RT: la corrida RT de rt_instability_epb.py con
     shunt de capa E (sigma = 1 + R: misma linea base, solo se anade la
     carga) se SUPRIME: con R=3 (F_s = 0.25) el drive diluido cae por debajo
     del amortiguamiento numerico fijo y el modo se ESTABILIZA (gamma < 0).
     El shunt cruza el umbral de inestabilidad: el mecanismo dia/noche de la
     EPB.

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 python elayer_epb.py
"""

import sys
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_twofluid import (
    EPBPhysParams, hll_flux, stack_state, unstack_state,
)
from PyExner.solvers.kernels.epb_sources import (
    EPBSourceParams, implicit_source_solve, solve_phi, solve_phi_sigma,
    _div_backward, _grad_forward, _face_avg, _poisson_divergence,
)
from PyExner.state.epb_twofluid_state import EPBTwoFluidState


# --------------------------------------------------------------------------- #
# Estado de prueba con corrientes no triviales (para A-C)                      #
# --------------------------------------------------------------------------- #

NX = NZ = 64
L = 2.0
DX = L / NX
N_ITER = 300


def _test_arr():
    """Estado suave doblemente periodico con div J != 0 (driver del solve)."""
    x = (np.arange(NX) + 0.5) * DX
    z = (np.arange(NZ) + 0.5) * DX
    X, Z = np.meshgrid(x, z, indexing="xy")
    kx = 2.0 * np.pi / L
    n = 1.0 + 0.3 * np.sin(kx * X) * np.cos(2 * kx * Z)
    jix = 0.20 * np.cos(kx * X)
    jiz = 0.10 * np.sin(2 * kx * Z) * np.cos(kx * X)
    jex = -0.15 * np.sin(kx * Z)
    jez = 0.05 * np.cos(kx * X + kx * Z)
    zero = np.zeros_like(n)
    arr = np.stack([n, n, jix, zero, jiz, jex, zero, jez], axis=-1)
    return jnp.asarray(arr)


def _sigma_var():
    """Conductancia no uniforme, positiva, de media ~1."""
    x = (np.arange(NX) + 0.5) * DX
    z = (np.arange(NZ) + 0.5) * DX
    X, Z = np.meshgrid(x, z, indexing="xy")
    kx = 2.0 * np.pi / L
    return jnp.asarray(1.0 + 0.6 * np.sin(kx * X) * np.sin(kx * Z) + 0.2 * np.cos(2 * kx * Z))


def test_A_equivalence():
    arr = _test_arr()
    phi0 = np.asarray(solve_phi(arr, DX, N_ITER))
    phi1 = np.asarray(solve_phi_sigma(arr, jnp.ones((NZ, NX)), DX, N_ITER))
    scale = max(np.abs(phi0).max(), 1e-300)
    err = np.abs(phi1 - phi0).max() / scale
    ok = err < 1e-9
    print(f"[A] equivalencia sigma=1 vs solve_phi: err_rel = {err:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_B_uniform_shunt():
    arr = _test_arr()
    ok = True
    phi1 = np.asarray(solve_phi_sigma(arr, jnp.ones((NZ, NX)), DX, N_ITER))
    scale = max(np.abs(phi1).max(), 1e-300)
    for R in (1.0, 3.0, 9.0):
        c = 1.0 + R
        phic = np.asarray(solve_phi_sigma(arr, jnp.full((NZ, NX), c), DX, N_ITER))
        err = np.abs(phic - phi1 / c).max() / scale
        good = err < 1e-9
        ok &= good
        print(f"[B] shunt uniforme R={R:.0f}: |phi_c - phi_1/(1+R)| = {err:.3e}  "
              f"{'PASS' if good else 'FAIL'}")
    return ok


def test_C_variable_cleaning():
    arr = _test_arr()
    sigma = _sigma_var()
    phi = solve_phi_sigma(arr, sigma, DX, N_ITER)
    rhs = _poisson_divergence(arr, DX)
    # corriente de correccion en caras: sigma_f G^+ phi
    fx = _face_avg(sigma, 1) * _grad_forward(phi, DX, axis=1)
    fz = _face_avg(sigma, 0) * _grad_forward(phi, DX, axis=0)
    div_clean = rhs - (_div_backward(fx, DX, axis=1) + _div_backward(fz, DX, axis=0))
    res = float(jnp.abs(div_clean).max() / jnp.abs(rhs).max())
    # El CG congela en residual relativo L2 < 1e-12; en norma-max el residuo
    # heredado puede ser ~1e-10 (vs 9.8e-2 del esquema central viejo).
    ok = res < 1e-8
    print(f"[C] limpieza con sigma variable: |D^-J - L_sigma phi| / |D^-J| = "
          f"{res:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# D) Apantallamiento de la RT (corrida RT base con cierre sigma = 1 + R)       #
# --------------------------------------------------------------------------- #

PHYS = EPBPhysParams(Ti=0.0, Te=0.0)
LX = LZ = 4.0
NXR = NZR = 64
DXR = LX / NXR
G = 1.0
B = 1.0
N_LOW, N_HIGH = 0.2, 1.0
Z_C, WIDTH, EPS, KMODE = LZ / 2.0, 0.3, 1e-2, 2
V_CHAR = max(PHYS.Mi, PHYS.Me) * G / (PHYS.e * B)
DT = 0.3 * DXR / max(V_CHAR, 1e-3)
NSTEPS = 320


def _build_rt_state():
    xc = (np.arange(NXR) + 0.5) * DXR
    zc = (np.arange(NZR) + 0.5) * DXR
    X, Z = np.meshgrid(xc, zc, indexing="xy")
    base = N_LOW + (N_HIGH - N_LOW) * np.exp(-((Z - Z_C) ** 2) / (2.0 * WIDTH**2))
    env = np.exp(-((Z - (Z_C - WIDTH)) ** 2) / (2.0 * WIDTH**2))
    pert = 1.0 + EPS * np.cos(2.0 * np.pi * KMODE * X / LX) * env
    n = jnp.asarray(base * pert)
    z = jnp.zeros((NZR, NXR))
    return EPBTwoFluidState(n_i=n, n_e=n, j_ix=z, j_iy=z, j_iz=z,
                            j_ex=z, j_ey=z, j_ez=z)


def _transport(Q, dt, dx, phys):
    Fp = hll_flux(Q, jnp.roll(Q, -1, axis=1), phys, "x")
    Gp = hll_flux(Q, jnp.roll(Q, -1, axis=0), phys, "z")
    return Q - (dt / dx) * ((Fp - jnp.roll(Fp, 1, axis=1)) +
                            (Gp - jnp.roll(Gp, 1, axis=0)))


def _amp(state):
    n = np.asarray(state.n_i)
    return float(np.sum((n - n.mean(axis=1, keepdims=True)) ** 2))


def _growth_rate(amps, dt):
    n = len(amps)
    lo, hi = n // 8, n // 2
    t = np.arange(lo, hi) * dt
    y = 0.5 * np.log(amps[lo:hi])
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope


def _run_rt(R_shunt: float):
    src = EPBSourceParams(gz=-G, By=B)
    state0 = _build_rt_state()
    sigma = jnp.ones((NZR, NXR)) + R_shunt   # Sigma_F uniforme (linea base de
    # rt_instability_epb.py) + shunt: experimento CONTROLADO, solo cambia la carga E.

    def _step(s):
        Q = _transport(stack_state(s), DT, DXR, PHYS)
        s = unstack_state(Q)
        return implicit_source_solve(s, DT, PHYS, src, DXR, sigma=sigma)

    step = jax.jit(_step)
    state = state0
    amps = [_amp(state)]
    for _ in range(NSTEPS):
        state = step(state)
        amps.append(_amp(state))
    return np.array(amps)


def test_D_rt_shielding():
    print(f"[D] RT con shunt de capa E: malla {NXR}x{NZR}, dt={DT:.4f}, "
          f"{NSTEPS} pasos, sigma = 1 + R (linea base RT + shunt)")
    results = {}
    for R in (0.0, 3.0):
        amps = _run_rt(R)
        gam = _growth_rate(amps, DT)
        fac = amps[-1] / amps[0]
        results[R] = (gam, fac)
        Fs = 1.0 / (1.0 + R)
        print(f"    R={R:.0f}  F_s={Fs:.2f}:  gamma = {gam:.3f}   "
              f"A_N/A_0 = {fac:.3e}")
    g0, f0 = results[0.0]
    g3, f3 = results[3.0]
    ok = True
    # El drive ideal se diluye ~sqrt(F_s) y compite con el amortiguamiento
    # numerico FIJO del esquema (upwind HLL): gamma_net = sqrt(F_s)*gamma_ideal
    # - gamma_num. Con F_s = 0.25 el drive cae por debajo del amortiguamiento y
    # el modo se ESTABILIZA (gamma < 0): el shunt cruza el umbral, exactamente
    # el mecanismo dia/noche de la EPB.
    if not (g0 > 0.15):
        print(f"[D][FAIL] linea base no inestable (gamma0 = {g0:.3f})"); ok = False
    elif not (g3 < 0.0):
        print(f"[D][FAIL] el shunt R=3 no estabiliza (gamma = {g3:.3f})"); ok = False
    else:
        print(f"[D][PASS] el shunt cruza el umbral: gamma(R=0) = {g0:.3f} > 0 "
              f"(inestable) -> gamma(R=3) = {g3:.3f} < 0 (estabilizado)")
    if not (f3 < f0 / 3.0):
        print("[D][FAIL] la amplitud no se suprime con R=3"); ok = False
    else:
        print(f"[D][PASS] amplitud suprimida: {f3:.2e} << {f0:.2e} "
              f"(factor {f0 / max(f3, 1e-30):.1f}x)")
    return ok


def main():
    print("=== Capa E como carga de conductancia (shunt) ===\n")
    ok = True
    ok &= test_A_equivalence()
    ok &= test_B_uniform_shunt()
    ok &= test_C_variable_cleaning()
    print()
    ok &= test_D_rt_shielding()
    print("\n=== VEREDICTO:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
