"""Adveccion pura con solucion analitica exacta.

Aisla el nucleo del transporte hiperbolico (``physical_flux`` + ``hll_flux``)
sin contornos de por medio, usando un dominio PERIODICO (via ``jnp.roll``) y
una configuracion fisica que admite solucion cerrada:

PLASMA FRIO (isotermo con T_i = T_e = 0  =>  c_i = c_e = 0). Sin presion, el
sistema isotermo de dos fluidos se reduce a dinamica de gases sin presion. Con
una velocidad uniforme v_0 (corriente j = e n v_0), la densidad satisface

    d_t n + v_0 d_x n = 0   =>   n(x, t) = n_0(x - v_0 t),

es decir, TRASLACION PURA a velocidad v_0. El HLL con c=0 degenera a upwind, que
preserva exactamente v = v_0 (misma difusion numerica en continuidad y momento),
de modo que el unico error frente a la solucion continua es la difusion de
primer orden del upwind. Esto da:

  (1) Test de SIGNO en regimen dinamico: v_i = +j_i/(e n_i) y v_e = -j_e/(e n_e).
      Con j_ix = j_ex = +e n v_0, el pulso IONICO viaja a +v_0 (derecha) y el
      ELECTRONICO a -v_0 (izquierda). Discrimina la convencion de carga.
  (2) Test de ORDEN: refinando la malla, el error L1 frente a n_0(x - v_0 t)
      debe converger a orden ~1 (upwind).

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 python advection_epb.py
"""

import math
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_twofluid import (
    EPBPhysParams, hll_flux, stack_state, unstack_state,
)
from PyExner.state.epb_twofluid_state import EPBTwoFluidState, EPB_FIELD_ORDER


# Plasma frio: sin presion -> adveccion pura.
PHYS = EPBPhysParams(Ti=0.0, Te=0.0)

L = 1.0            # longitud del dominio periodico
N0 = 1.0           # densidad de fondo
AMP = 0.3          # amplitud del pulso gaussiano
SIGMA = 0.05       # ancho del pulso
V0 = 1.0           # velocidad de adveccion uniforme
CFL = 0.4
T_FINAL = 0.5      # con L=1, v0=1 -> medio dominio recorrido


# --------------------------------------------------------------------------- #
# Solucion exacta y condicion inicial                                          #
# --------------------------------------------------------------------------- #

def _periodic_gauss(x, center, shift):
    """Gaussiana sobre dominio periodico [0, L) desplazada ``shift``."""
    d = x - center - shift
    d = (d + 0.5 * L) % L - 0.5 * L   # distancia periodica con signo
    return N0 + AMP * np.exp(-d**2 / (2.0 * SIGMA**2))


def _build_state(nx):
    dx = L / nx
    xc = (np.arange(nx) + 0.5) * dx      # centros de celda
    n_i = _periodic_gauss(xc, L / 2, 0.0)
    n_e = n_i.copy()

    e = PHYS.e
    # Velocidad uniforme v0 -> corriente = e n v0.
    # Ion: j_ix = +e n v0 -> v_i = +v0 (derecha).
    # Electron: j_ex = +e n v0 -> v_e = -j/(e n) = -v0 (izquierda).
    j_ix = e * n_i * V0
    j_ex = e * n_e * V0
    z = np.zeros(nx)

    arr = np.stack([n_i, n_e, j_ix, z, z, j_ex, z, z], axis=-1)  # (nx, 8)
    return jnp.asarray(arr[None, :, :]), xc, dx   # (1, nx, 8)


# --------------------------------------------------------------------------- #
# Paso de transporte periodico (1D en x)                                       #
# --------------------------------------------------------------------------- #

def _periodic_step_x(Q, dt, dx, phys):
    """Volumen finito periodico: div_i = F_{i+1/2} - F_{i-1/2}."""
    QL = Q
    QR = jnp.roll(Q, -1, axis=1)            # vecino derecho (i+1), con wrap
    Fp = hll_flux(QL, QR, phys, "x")        # flujo en i+1/2
    Fm = jnp.roll(Fp, 1, axis=1)            # flujo en i-1/2
    return Q - (dt / dx) * (Fp - Fm)


def _run(nx):
    Q, xc, dx = _build_state(nx)
    dt = CFL * dx / V0
    nsteps = int(round(T_FINAL / dt))
    dt = T_FINAL / nsteps                   # ajustar para caer exacto en T_FINAL

    step = jax.jit(lambda Q: _periodic_step_x(Q, dt, dx, PHYS))
    for _ in range(nsteps):
        Q = step(Q)

    n_i = np.asarray(Q[0, :, 0])
    n_e = np.asarray(Q[0, :, 1])
    return n_i, n_e, xc, dx


# --------------------------------------------------------------------------- #
# Pruebas                                                                      #
# --------------------------------------------------------------------------- #

def test_sign_directions():
    """El pulso ionico se desplaza a +v0 y el electronico a -v0."""
    nx = 400
    n_i, n_e, xc, _ = _run(nx)

    # Centro de masa de la perturbacion (n - N0), con peso periodico via fase.
    def _phase_center(field):
        w = np.clip(field - N0, 0.0, None)
        theta = 2.0 * np.pi * xc / L
        cx = np.sum(w * np.cos(theta))
        cy = np.sum(w * np.sin(theta))
        ang = np.arctan2(cy, cx) % (2.0 * np.pi)
        return ang * L / (2.0 * np.pi)

    ci = _phase_center(n_i)
    ce = _phase_center(n_e)
    expected_i = (L / 2 + V0 * T_FINAL) % L      # 0.5 + 0.5 = 1.0 -> 0.0
    expected_e = (L / 2 - V0 * T_FINAL) % L      # 0.5 - 0.5 = 0.0

    ok = True
    di = min(abs(ci - expected_i), L - abs(ci - expected_i))
    de = min(abs(ce - expected_e), L - abs(ce - expected_e))
    print(f"[sign] ion   centro={ci:.4f}  esperado={expected_i:.4f}  err={di:.4f}")
    print(f"[sign] elec  centro={ce:.4f}  esperado={expected_e:.4f}  err={de:.4f}")
    if di > 0.02:
        print("[FAIL] el pulso ionico no viaja a +v0"); ok = False
    if de > 0.02:
        print("[FAIL] el pulso electronico no viaja a -v0"); ok = False
    if ok:
        print("[PASS] signos de adveccion correctos (ion +v0, electron -v0)")
    return ok


def test_convergence_order():
    """El error L1 frente a la solucion exacta converge a orden ~1 (upwind)."""
    resolutions = [50, 100, 200, 400]
    errs = []
    for nx in resolutions:
        n_i, _, xc, dx = _run(nx)
        exact = _periodic_gauss(xc, L / 2, V0 * T_FINAL)
        err = float(np.sum(np.abs(n_i - exact)) * dx)   # norma L1
        errs.append(err)

    print("\n[convergencia] error L1 vs solucion exacta n0(x - v0 t):")
    orders = []
    for k in range(len(resolutions)):
        if k == 0:
            print(f"  Nx={resolutions[k]:4d}  L1={errs[k]:.3e}")
        else:
            p = math.log(errs[k - 1] / errs[k]) / math.log(resolutions[k] / resolutions[k - 1])
            orders.append(p)
            print(f"  Nx={resolutions[k]:4d}  L1={errs[k]:.3e}  orden={p:.3f}")

    mean_order = sum(orders) / len(orders)
    print(f"[convergencia] orden medio = {mean_order:.3f}")
    # Upwind de primer orden: aceptamos [0.7, 1.3] (la difusion + suavidad del
    # pulso pueden sesgar ligeramente el orden asintotico).
    ok = 0.7 <= mean_order <= 1.3 and all(errs[k] < errs[k - 1] for k in range(1, len(errs)))
    print("[PASS] orden de convergencia ~1 (upwind)" if ok
          else "[FAIL] el orden de convergencia no es ~1 o el error no decrece")
    return ok


def test_positivity_and_mass():
    """Positividad y conservacion EXACTA de masa (dominio periodico)."""
    nx = 200
    n_i, n_e, xc, dx = _run(nx)
    ok = True
    if n_i.min() <= 0 or n_e.min() <= 0:
        print(f"[FAIL] positividad: min n_i={n_i.min():.4f}, n_e={n_e.min():.4f}"); ok = False
    # Masa inicial analitica vs final numerica.
    n_i0 = _periodic_gauss(xc, L / 2, 0.0)
    m0 = float(np.sum(n_i0) * dx)
    mN = float(np.sum(n_i) * dx)
    rel = abs(mN - m0) / m0
    print(f"[masa] periodica  m0={m0:.6f}  mN={mN:.6f}  drift={rel*100:.2e}%")
    if rel > 1e-10:
        print("[FAIL] la masa no se conserva en dominio periodico"); ok = False
    if ok:
        print("[PASS] positividad + conservacion exacta de masa")
    return ok


if __name__ == "__main__":
    print("=== Adveccion pura (plasma frio, dominio periodico) ===")
    results = [
        test_sign_directions(),
        test_convergence_order(),
        test_positivity_and_mass(),
    ]
    npass = sum(results)
    print(f"\n{npass}/{len(results)} pruebas OK")
    raise SystemExit(0 if npass == len(results) else 1)
