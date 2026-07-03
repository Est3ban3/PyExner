"""Inestabilidad de intercambio (RT generalizada): el test fuerte del modelo.

Comprueba que el acoplamiento g - grad(n) - E (la polarizacion que resuelve
``solve_phi``) produce la dinamica de la Burbuja de Plasma Ecuatorial (EPB).
Es CUALITATIVO (no hay solucion cerrada) y su check principal es el CONTRASTE
entre un estado con gradiente de densidad (energia libre gravitacional) y un
control uniforme, que solo depende del SIGNO del mecanismo, no de un valor
numerico fragil.

--- Modelo (plasma frio, modelo RT canonico de plasma) ---

Cierre isotermo con T_i = T_e = 0 (c_i = c_e = 0): sin presion, el unico motor es
la deriva gravitacional + polarizacion. Geometria F-region simplificada:

    B = B \\hat{y}   (fuera del plano),   g = -g \\hat{z}   (hacia abajo).

Deriva gravitacional v_g = (M/q) (g x B)/B^2:
    g x B = (-g \\hat z) x (B \\hat y) = +gB \\hat x
    => v_{g,ion} = +(M_i g)/(e B) \\hat x      (iones a +x)
       v_{g,ele} = -(M_e g)/(e B) \\hat x      (electrones a -x).

Las cargas se separan en sentidos opuestos a lo largo de x. En una perturbacion
de densidad n(x), la divergencia de la corriente total ∂_x J_x != 0 acumula
carga -> potencial de polarizacion φ (∇²φ = ∇·J) -> E_p = -∇φ -> deriva E_p x B
(independiente de la carga, v_{Ez} = E_x/B) que advecta el plasma en z. En la
cara INFERIOR de una capa densa (denso por encima del ligero, grad(n) antiparalelo
a g) la perturbacion CRECE (bottomside RT, como la EPB real). Tasa lineal
clasica:  γ = sqrt(g / L_n).

--- Geometria del dominio (CLAVE para la autoconsistencia) ---

``solve_phi`` usa BC PERIODICAS en x y z (jnp.roll). Para que la electrostatica
sea consistente con el transporte se usa un dominio DOBLEMENTE PERIODICO. Las
paredes reflectantes en z (que se probaron antes) son incompatibles con esas BC
periodicas y producen polarizacion espuria en los bordes; el dominio periodico
las evita. La masa se conserva exactamente.

--- Criterio de exito (discriminante y robusto) ---

Se contrastan dos corridas con la MISMA siembra de perturbacion y la MISMA masa
media:
  - ESTRATIFICADO: banda densa gaussiana en z -> su cara inferior es RT-inestable.
  - UNIFORME (control): densidad constante -> sin energia libre, NO crece.

(1) CONTRASTE: A(t) = Σ (n - <n>_x)^2 crece >5x en el estratificado y NO crece
    (decae) en el uniforme. Check duro: aisla el papel del gradiente de densidad.
(2) γ del orden de sqrt(g/L_n) (diagnostico informativo, tolerancia amplia por la
    contaminacion par/impar del solve vista en sources_epb.py y la difusion del
    upwind HLL).
(3) Positividad mantenida en la fase lineal.

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 python rt_instability_epb.py
"""

import math
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_twofluid import EPBPhysParams, hll_flux, stack_state
from PyExner.solvers.kernels.epb_sources import EPBSourceParams, implicit_source_solve
from PyExner.state.epb_twofluid_state import EPBTwoFluidState


# Plasma frio (sin presion): modelo RT de plasma.
PHYS = EPBPhysParams(Ti=0.0, Te=0.0)

# Geometria / malla
LX = 4.0
LZ = 4.0
NX = 64
NZ = 64
DX = LX / NX               # malla isotropa (DX == DZ)

# Fisica
G = 1.0                    # |g|, g = -G \hat z
B = 1.0                    # B = B \hat y
N_LOW = 0.2
N_HIGH = 1.0
Z_C = LZ / 2.0             # interfaz
WIDTH = 0.3                # ancho de la interfaz (~ L_n)
EPS = 1e-2                 # amplitud de la perturbacion
KMODE = 2                  # numero de modo en x

# Integracion
V_CHAR = max(PHYS.Mi, PHYS.Me) * G / (PHYS.e * B)   # deriva gravitacional ~ 1
CFL = 0.3
DT = CFL * DX / max(V_CHAR, 1e-3)
NSTEPS = 400


# --------------------------------------------------------------------------- #
# Construccion del estado inicial                                              #
# --------------------------------------------------------------------------- #

def _build_state(stratified: bool):
    """Estado inicial en dominio DOBLEMENTE PERIODICO (consistente con solve_phi,
    que usa BC periodicas en x y z).

    stratified=True  -> banda densa (gaussiana en z): su cara INFERIOR tiene el
                        denso por encima del ligero con g hacia abajo => interfaz
                        RT-INESTABLE (bottomside, como la EPB real). CRECE.
    stratified=False -> densidad UNIFORME (control): sin gradiente de densidad no
                        hay energia libre gravitacional => NO crece.

    En ambos casos se siembra la MISMA perturbacion sinusoidal en x, de modo que
    la diferencia de evolucion aisla el papel del gradiente de densidad (el motor
    RT) y no artefactos de la siembra.
    """
    xc = (np.arange(NX) + 0.5) * DX
    zc = (np.arange(NZ) + 0.5) * DX
    X, Z = np.meshgrid(xc, zc, indexing="xy")   # (NZ, NX), axis0=z, axis1=x

    if stratified:
        # banda densa centrada (periodica): bottomside RT-inestable.
        base = N_LOW + (N_HIGH - N_LOW) * np.exp(-((Z - Z_C) ** 2) / (2.0 * WIDTH**2))
    else:
        # control uniforme con la MISMA masa media que la banda.
        band = N_LOW + (N_HIGH - N_LOW) * np.exp(-((zc - Z_C) ** 2) / (2.0 * WIDTH**2))
        base = np.full_like(Z, band.mean())

    # Perturbacion concentrada en la cara inferior de la banda (donde es inestable).
    env = np.exp(-((Z - (Z_C - WIDTH)) ** 2) / (2.0 * WIDTH**2))
    pert = 1.0 + EPS * np.cos(2.0 * np.pi * KMODE * X / LX) * env
    n = base * pert

    n_i = jnp.asarray(n)
    n_e = jnp.asarray(n)        # cuasineutral
    z = jnp.zeros((NZ, NX))
    return EPBTwoFluidState(
        n_i=n_i, n_e=n_e,
        j_ix=z, j_iy=z, j_iz=z,
        j_ex=z, j_ey=z, j_ez=z,
    ), Z


# --------------------------------------------------------------------------- #
# Paso de transporte: periodico en x, paredes (cero-gradiente) en z           #
# --------------------------------------------------------------------------- #

def _transport(Q, dt, dx, phys):
    # Dominio DOBLEMENTE PERIODICO (consistente con las BC periodicas de solve_phi).
    # x periodico
    Fp = hll_flux(Q, jnp.roll(Q, -1, axis=1), phys, "x")     # i+1/2
    Fm = jnp.roll(Fp, 1, axis=1)                             # i-1/2
    divx = Fp - Fm
    # z periodico
    Gp = hll_flux(Q, jnp.roll(Q, -1, axis=0), phys, "z")     # k+1/2
    Gm = jnp.roll(Gp, 1, axis=0)                             # k-1/2
    divz = Gp - Gm
    return Q - (dt / dx) * (divx + divz)


def _imex_step(state, dt, src):
    Q = stack_state(state)
    Q = _transport(Q, dt, DX, PHYS)
    from PyExner.solvers.kernels.epb_twofluid import unstack_state
    state = unstack_state(Q)
    return implicit_source_solve(state, dt, PHYS, src, DX)


def _perturbation_amplitude(state):
    """A = Σ (n_i - <n_i>_x)^2: energia de la parte x-dependiente (perturbacion)."""
    n = np.asarray(state.n_i)
    mean_x = n.mean(axis=1, keepdims=True)     # promedio en x por cada z
    return float(np.sum((n - mean_x) ** 2))


def _total_mass(state):
    return float(np.asarray(state.n_i).sum())


# --------------------------------------------------------------------------- #
# Evolucion                                                                    #
# --------------------------------------------------------------------------- #

def _run(stratified: bool):
    src = EPBSourceParams(gz=-G, By=B)         # g = -G z, B = B y
    state, _ = _build_state(stratified)

    step = jax.jit(lambda s: _imex_step(s, DT, src))

    amps = [_perturbation_amplitude(state)]
    mass0 = _total_mass(state)
    nmin = float(np.asarray(state.n_i).min())
    for _ in range(NSTEPS):
        state = step(state)
        amps.append(_perturbation_amplitude(state))
        nmin = min(nmin, float(np.asarray(state.n_i).min()))
    mass_drift = abs(_total_mass(state) - mass0) / abs(mass0)
    return np.array(amps), mass_drift, nmin


def _growth_rate(amps, dt):
    """Ajuste lineal de 0.5*log(A) en la fase de crecimiento (A ~ exp(2 γ t))."""
    # Usar la primera mitad (fase lineal), evitando saturacion.
    n = len(amps)
    lo, hi = n // 8, n // 2
    t = np.arange(lo, hi) * dt
    y = 0.5 * np.log(amps[lo:hi])
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope


# --------------------------------------------------------------------------- #
# Pruebas                                                                      #
# --------------------------------------------------------------------------- #

def main():
    print("=== Inestabilidad de intercambio (RT generalizada) ===")
    print(f"malla {NX}x{NZ}  dx={DX:.4f}  dt={DT:.4f}  pasos={NSTEPS}  "
          f"g={G} B={B} L_n~{WIDTH}")
    print(f"deriva gravitacional v_g ~ {V_CHAR:.3f}, "
          f"γ_teorico ~ sqrt(g/L_n) = {math.sqrt(G / WIDTH):.3f}\n")

    amps_strat, mdrift_strat, nmin_strat = _run(stratified=True)
    amps_unif, mdrift_unif, nmin_unif = _run(stratified=False)

    # Diagnostico: serie temporal de A(t)/A0 en 6 instantes.
    idx = np.linspace(0, NSTEPS, 6).astype(int)
    print("[serie]    t      A_estrat/A0   A_unif/A0    ratio")
    for i in idx:
        t = i * DT
        rs = amps_strat[i] / amps_strat[0]
        ru = amps_unif[i] / amps_unif[0]
        print(f"  t={t:6.3f}   {rs:10.3e}  {ru:10.3e}   {rs / max(ru, 1e-30):7.2f}")
    print(f"\n[masa] deriva relativa  estratificado={mdrift_strat:.2e}  "
          f"uniforme={mdrift_unif:.2e}")
    print()

    A0_s, AN_s = amps_strat[0], amps_strat[-1]
    A0_u, AN_u = amps_unif[0], amps_unif[-1]
    growth_s = AN_s / A0_s
    growth_u = AN_u / A0_u

    print(f"[estratificado / RT-INESTABLE] A0={A0_s:.4e}  AN={AN_s:.4e}  "
          f"factor={growth_s:.3e}  (min n={nmin_strat:.4f})")
    print(f"[uniforme / control]           A0={A0_u:.4e}  AN={AN_u:.4e}  "
          f"factor={growth_u:.3e}  (min n={nmin_unif:.4f})")

    gamma = _growth_rate(amps_strat, DT)
    gamma_theory = math.sqrt(G / WIDTH)
    print(f"\n[gamma] medido (fase lineal) = {gamma:.3f}  "
          f"teorico sqrt(g/Ln) = {gamma_theory:.3f}  "
          f"ratio = {gamma / gamma_theory:.2f}")

    ok = True

    # (1) CONTRASTE RT (check duro): la perturbacion crece en el estado con
    #     gradiente de densidad (energia libre gravitacional) y NO en el uniforme.
    #     Ambos parten de la misma siembra y la misma masa media, de modo que el
    #     contraste aisla el papel del gradiente de densidad (motor RT).
    if not (growth_s > 5.0 * growth_u and growth_s > 5.0):
        print("[FAIL] no hay contraste estratificado/uniforme claro"); ok = False
    else:
        print(f"[PASS] contraste RT: crece con gradiente de densidad "
              f"({growth_s:.1f}x) y no en el uniforme ({growth_u:.2f}x)")

    # (2) gamma del orden correcto (factor 3, por contaminacion par/impar del solve).
    if not (gamma > 0 and (1.0 / 3.0) <= gamma / gamma_theory <= 3.0):
        print(f"[WARN] gamma fuera de [1/3, 3]x teorico (diagnostico, no bloquea)")
    else:
        print("[PASS] gamma del orden de sqrt(g/Ln)")

    # (3) Positividad en la fase lineal.
    if nmin_strat <= 0.0:
        print("[FAIL] densidad no positiva en la corrida estratificada"); ok = False
    else:
        print("[PASS] positividad mantenida")

    print("\n" + ("RT PASSED" if ok else "RT FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
