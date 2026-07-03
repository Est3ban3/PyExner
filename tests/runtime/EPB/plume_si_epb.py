"""Pluma EPB en unidades FISICAS (SI): la figura clasica altitud vs zonal.

Corrida 2D no lineal del modelo completo (transporte MUSCL + fuentes rigidas
implicitas + solve sigma-variable + quimica) sobre un dominio ecuatorial
realista:

    distancia zonal x en [0, 400] km,   altitud z en [200, 800] km,

con capa F2 Chapman ELEVADA post-PRE (h_mF2 = 450 km, el escenario de
pre_onset_epb.py). Produce ``figures/fig5_pluma_si.png``: mapas de
log10 n_e(x, z) en 4 instantes — la grafica tipica de las simulaciones de EPB
(depleciones del bottomside ascendiendo como plumas hacia el topside).

--- Por que es viable en SI (sin el CFL electronico) ---

Cierre FRIO (T_i = T_e = 0): no hay velocidades acusticas (c_e ~ 120 km/s
desaparece del HLL); las velocidades caracteristicas del transporte son las
DERIVAS (~10-500 m/s). La rigidez de giracion (Omega_i ~ 180 s^-1,
Omega_e ~ 5e6 s^-1) vive en el solve IMPLICITO 6x6: backward Euler con
Omega dt >> 1 entrega exactamente las derivas estacionarias j = -A^{-1} b
(E x B + Pedersen + gravitacional) — un cierre de derivas de facto, A-estable.
dt queda limitado solo por el CFL de las derivas: dt ~ 2-6 s.

--- Cierre electrostatico (clave dimensional y de estabilidad) ---

El potencial es DIAGNOSTICO de la densidad (cierre tipo Ossakow): el RHS del
solve sigma-variable es SOLO la corriente motriz (gravitacional),
no la corriente total del estado:

    div( sigma_P grad phi ) = div J_g ,   J_g = (n M_i g / B) x_hat ,

con sigma_P = n e^2 nu_in / (M_i (nu_in^2 + Omega_i^2)) [S/m] => phi [V].
Luego E = -grad phi se pasa CONGELADO al solve implicito 6x6 (E_ext), que
entrega las derivas estacionarias consistentes (E x B + Pedersen + grav).
Balance local: sigma_P E_x ~ -delta(J_g)  =>  v_z = E_x/B = (dn/n) g/nu_in,
el resultado RT colisional exacto; el campo se intensifica dentro de la
deplecion (sigma_P propto n), la fisica que acelera la pluma.

POR QUE NO con la corriente total: el estado tras el solve implicito ya
contiene la respuesta Pedersen sigma_P E^n; re-derivar phi^{n+1} de esa
corriente da phi^{n+1} ~ phi_eq - phi^n (lazo lagged marginal) que la no
linealidad vuelve inestable (~x5/paso con dt = 6 s, verificado). Con el
cierre drive-driven phi no depende de j y el unico acoplamiento es el
transporte de n (gamma_RT ~ 1e-2 s^-1 << 1/dt): estable.

--- Honestidad ---

- nu_in ESCALAR (valor del bottomside, ~0.05 s^-1 a ~420 km): exagera la
  friccion en el topside (pluma algo lenta arriba); nu_in(z) como campo en el
  solve implicito queda pendiente.
- BC periodicas en x (fisico: onda zonal) y en z (artificio): el wrap
  topside->bottomside es estable RT (denso abajo en ambos lados del wrap) y
  la quimica ancla el fondo; la ventana de interes queda lejos de los bordes.
- Sin E0 de fondo (marco co-movil con la subida E x B del PRE).

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 JAX_PLATFORMS=cpu python plume_si_epb.py
"""

import os
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_twofluid import EPBPhysParams
from PyExner.solvers.kernels.epb_sources import (
    EPBSourceParams, implicit_source_solve, electric_field, _poisson_cg_sigma,
)
from PyExner.solvers.kernels.epb_fluxtube import pedersen_conductivity
from PyExner.solvers.kernels.epb_ionosphere import (
    SI_CONST, M_OPLUS, FRegionParams,
    chapman_layer, exp_profile, dipole_B, gravity, chemistry_step_state,
)
from PyExner.state.epb_twofluid_state import EPBTwoFluidState

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dominio fisico (malla ISOTROPA: dx = dz, requerida por los kernels)          #
# --------------------------------------------------------------------------- #

Z0, Z1 = 200.0e3, 800.0e3          # altitud [m]
LX = 400.0e3                       # extension zonal [m]
NX = 768
DX = LX / NX                       # 4166.7 m
NZ = int(round((Z1 - Z0) / DX))    # 144  ->  LZ = 600 km exacto

# --------------------------------------------------------------------------- #
# Fondo ionosferico: capa ELEVADA post-PRE (escenario de pre_onset_epb.py)     #
# --------------------------------------------------------------------------- #

FR = FRegionParams(h_peak=450.0e3)
N_FLOOR = 5.0e-3 * FR.n_max        # piso de densidad (suaviza el wrap en z)

H_REF = 420.0e3                    # altura de referencia (bottomside elevado)
B0 = float(dipole_B(jnp.asarray(H_REF), FR.B_surf_eq))     # ~2.4e-5 T
G0 = float(gravity(jnp.asarray(H_REF)))                    # ~8.6 m/s^2
NU_IN = 0.05                       # nu_in(~420 km) [s^-1]
NU_EN = 0.005

PHYS = EPBPhysParams(e=SI_CONST.e, kB=SI_CONST.kB, Mi=M_OPLUS,
                     Me=SI_CONST.Me, Ti=0.0, Te=0.0)        # plasma FRIO
SRC = EPBSourceParams(gz=-G0, By=B0, nu_in=NU_IN, nu_en=NU_EN,
                      poisson_iters=800)

N_REF = FR.n_max
N_HARD = 1.0e-3 * FR.n_max         # piso DURO post-transporte (positividad MUSCL)
# Piso de conductividad: en 2D local sigma_P -> 0 dentro de la deplecion
# amplifica E sin cota (v ~ (n_bg/n_piso) g/nu_in: con piso 2% se midieron
# ~2 km/s y turbulencia saturada). En 3D la conductancia INTEGRADA del tubo
# (topside + capa E conjugada) acota esa amplificacion; el piso al 25% de
# n_max es su sustituto 2D y limita v_burbuja a ~2-4 x g/nu_in ~ 400 m/s,
# el rango observado (100-500 m/s).
SIGMA_MIN = float(pedersen_conductivity(0.25 * FR.n_max, NU_IN, B0))

# Difusion sub-grid en continuidad: en el limite RT local gamma ~ g/(nu Ln)
# es independiente de k => los modos de REJILLA crecen mas rapido que las
# plumas y el RHS centrado no les genera respuesta de phi (aniquila Nyquist).
# D ~ 0.15 v_ref dx corta la rejilla (tau_grid ~ 15 s) con Peclet ~ 500 en
# la escala de las plumas (~50 km): estandar en codigos EPB (tipo Zalesak).
V_REF = G0 / NU_IN                                   # ~173 m/s
D_NUM = 0.15 * V_REF * (LX / NX)                     # ~1.1e5 m^2/s

# --------------------------------------------------------------------------- #
# Estado inicial: Chapman + piso, semilla multi-modo en el bottomside          #
# --------------------------------------------------------------------------- #

zc = Z0 + (np.arange(NZ) + 0.5) * DX
xc = (np.arange(NX) + 0.5) * DX
X, Z = np.meshgrid(xc, zc, indexing="xy")                  # (NZ, NX)

n0_z = np.asarray(chapman_layer(jnp.asarray(zc), FR.n_max, FR.h_peak,
                                FR.H_chapman)) + N_FLOOR
beta_z = np.asarray(exp_profile(jnp.asarray(zc), FR.beta0, FR.h_ref, FR.H_beta))
PROD = jnp.asarray((beta_z * n0_z)[:, None])               # P = beta n0 (ancla)
BETA = jnp.asarray(beta_z[:, None])

env = np.exp(-((Z - (FR.h_peak - 1.5 * FR.H_chapman)) ** 2) / (2.0 * (30.0e3) ** 2))
seed = 1.0 - 0.05 * env * (np.cos(2.0 * np.pi * 2.0 * X / LX)
                           + 0.6 * np.cos(2.0 * np.pi * 3.0 * X / LX + 1.9))
n_init = n0_z[:, None] * seed

_zeros = jnp.zeros((NZ, NX))
STATE0 = EPBTwoFluidState(
    n_i=jnp.asarray(n_init), n_e=jnp.asarray(n_init),
    j_ix=_zeros, j_iy=_zeros, j_iz=_zeros,
    j_ex=_zeros, j_ey=_zeros, j_ez=_zeros,
)

# --------------------------------------------------------------------------- #
# Paso temporal: continuidad MUSCL (derivas) -> phi(n) -> fuentes -> quimica   #
#                                                                              #
# Con Omega_i dt ~ 10^3 el backward-Euler regenera j = -A^{-1} b + O(1/Om dt)  #
# cada paso: la corriente es DIAGNOSTICA (cierre de derivas), y transportar    #
# su inercia es fisicamente irrelevante y numericamente fatal (en la           #
# reconstruccion MUSCL de momento, v_f = j_f/(e n_f) -> inf cuando n_f -> 0    #
# en la deplecion; explosion verificada en t ~ 350 s). Como en los codigos     #
# EPB clasicos (Ossakow/Zalesak), solo se advecta la CONTINUIDAD:              #
#                                                                              #
#     dn_s/dt + div(n_s v_s) = 0,   v_s = j_s / (q_s n_s)  (deriva por celda)  #
#                                                                              #
# Cada especie con SU deriva: la diferencia es la corriente, cuya divergencia  #
# anula el solve de phi => cuasi-neutralidad preservada al nivel del CG.       #
# --------------------------------------------------------------------------- #


def _mc_slope(n, axis):
    """Pendiente limitada MC (monotonized central) por eje, periodica."""
    dp = jnp.roll(n, -1, axis) - n
    dm = n - jnp.roll(n, 1, axis)
    s = jnp.sign(dp) + jnp.sign(dm)
    return 0.5 * s * jnp.minimum(jnp.minimum(jnp.abs(dp), jnp.abs(dm)),
                                 0.25 * jnp.abs(dp + dm))


def _advect_div(n, vx, vz):
    """div(n v) conservativa: MUSCL upwind con v en caras (media aritmetica)."""
    out = jnp.zeros_like(n)
    for v, ax in ((vx, 1), (vz, 0)):
        sl = _mc_slope(n, ax)
        nL = n + 0.5 * sl                       # cara i+1/2 desde la celda i
        nR = jnp.roll(n - 0.5 * sl, -1, ax)     # desde la celda i+1
        vf = 0.5 * (v + jnp.roll(v, -1, ax))    # velocidad en la cara i+1/2
        F = jnp.where(vf > 0.0, vf * nL, vf * nR)
        out = out + (F - jnp.roll(F, 1, ax)) / DX
    return out


def _lap(n):
    """Laplaciano compacto 5 puntos, periodico (difusion sub-grid)."""
    return (jnp.roll(n, -1, 0) + jnp.roll(n, 1, 0) + jnp.roll(n, -1, 1)
            + jnp.roll(n, 1, 1) - 4.0 * n) / DX**2


@jax.jit
def step(s, dt):
    e = PHYS.e
    n_i = jnp.maximum(s.n_i, N_HARD)
    n_e = jnp.maximum(s.n_e, N_HARD)
    vi_x, vi_z = s.j_ix / (e * n_i), s.j_iz / (e * n_i)
    ve_x, ve_z = -s.j_ex / (e * n_e), -s.j_ez / (e * n_e)
    n_i = jnp.maximum(n_i - dt * (_advect_div(n_i, vi_x, vi_z) - D_NUM * _lap(n_i)),
                      N_HARD)
    n_e = jnp.maximum(n_e - dt * (_advect_div(n_e, ve_x, ve_z) - D_NUM * _lap(n_e)),
                      N_HARD)
    s = s.replace(n_i=n_i, n_e=n_e)
    n = 0.5 * (n_i + n_e)
    sigma = jnp.maximum(pedersen_conductivity(n, NU_IN, B0), SIGMA_MIN)
    # Drive gravitacional J_g = (n M_i g / B) x_hat; divergencia centrada
    # (= D^- aplicado a la media de caras, consistente con L_sigma).
    Jg = n * (PHYS.Mi * G0 / B0)
    rhs = (jnp.roll(Jg, -1, axis=1) - jnp.roll(Jg, 1, axis=1)) / (2.0 * DX)
    phi = _poisson_cg_sigma(rhs, sigma, DX, SRC.poisson_iters)
    E = electric_field(phi, SRC, DX)
    s = implicit_source_solve(s, dt, PHYS, SRC, DX, E_ext=E)
    return chemistry_step_state(s, dt, PROD, BETA)


def _vmax(s):
    """Maxima velocidad de deriva |j|/(e n) sobre especies y componentes [m/s]."""
    e = PHYS.e
    ni = np.maximum(np.asarray(s.n_i), N_FLOOR)
    ne = np.maximum(np.asarray(s.n_e), N_FLOOR)
    vix = np.abs(np.asarray(s.j_ix)) / (e * ni)
    viz = np.abs(np.asarray(s.j_iz)) / (e * ni)
    vex = np.abs(np.asarray(s.j_ex)) / (e * ne)
    vez = np.abs(np.asarray(s.j_ez)) / (e * ne)
    return float(max(vix.max(), viz.max(), vex.max(), vez.max()))


def run():
    CFL = 0.2
    DT_MAX = 6.0
    V_FLOOR = 40.0
    T_FINAL = 2800.0                       # hasta penetracion del topside
    SNAP_T = [0.0, 1400.0, 2100.0, 2800.0]

    state = STATE0
    snaps = [(0.0, np.asarray(state.n_e))]
    next_snap = 1

    t, k = 0.0, 0
    dt = DT_MAX
    t_wall = time.time()
    while t < T_FINAL - 1e-9:
        if k % 5 == 0:
            vm = _vmax(state)
            if not np.isfinite(vm):
                raise FloatingPointError(f"v_max no finito en t={t:.1f} s (paso {k})")
            dt = min(CFL * DX / max(vm, V_FLOOR), DT_MAX)
        dt_eff = min(dt, T_FINAL - t,
                     (SNAP_T[next_snap] - t) if next_snap < len(SNAP_T) else 1e30)
        state = step(state, jnp.float64(dt_eff))
        t += dt_eff
        k += 1
        if k % 40 == 0:
            n = np.asarray(state.n_e)
            depth = float((n / n0_z[:, None]).min())
            print(f"  t={t:7.1f} s  dt={dt_eff:5.2f}  v_max={_vmax(state):7.1f} m/s  "
                  f"n_min={n.min():.2e}  deplecion min(n/n0)={depth:.3f}")
        if next_snap < len(SNAP_T) and t >= SNAP_T[next_snap] - 1e-6:
            snaps.append((t, np.asarray(state.n_e)))
            next_snap += 1
    print(f"[run] {k} pasos en {time.time() - t_wall:.1f} s de pared")

    # velocidad de ascenso del apice de la pluma (entre los 2 ultimos snapshots)
    interior = (zc > Z0 + 30.0e3) & (zc < Z1 - 30.0e3)   # excluye bandas del wrap

    def _apex(nk):
        depleted = ((nk / n0_z[:, None]) < 0.5) & interior[:, None]
        rows = np.where(depleted.any(axis=1))[0]
        return zc[rows[-1]] if len(rows) else zc[0]

    (t_a, n_a), (t_b, n_b) = snaps[-2], snaps[-1]
    v_apex = (_apex(n_b) - _apex(n_a)) / max(t_b - t_a, 1e-9)
    print(f"[diag] apice: {_apex(n_a) / 1e3:.0f} -> {_apex(n_b) / 1e3:.0f} km  "
          f"=> v_ascenso ~ {v_apex:.0f} m/s (obs: 100-500 m/s)")
    print(f"[diag] deplecion final: {(np.asarray(state.n_e) / n0_z[:, None]).min():.2e} "
          f"del fondo (obs: 1-3 ordenes)")
    return snaps


def plot(snaps):
    fig, axs = plt.subplots(1, len(snaps), figsize=(3.6 * len(snaps), 5.6),
                            sharey=True)
    vmin, vmax = 9.5, np.log10(1.3 * FR.n_max)
    for ax, (tk, nk) in zip(axs, snaps):
        im = ax.pcolormesh(xc / 1e3, zc / 1e3, np.log10(np.maximum(nk, 1.0)),
                           cmap="turbo", vmin=vmin, vmax=vmax, shading="auto")
        ax.axhline(FR.h_peak / 1e3, color="w", ls=":", lw=0.8, alpha=0.7)
        ax.set_title(f"t = {tk / 60.0:.0f} min")
        ax.set_xlabel("distancia zonal [km]")
    axs[0].set_ylabel("altitud [km]")
    axs[0].text(8, FR.h_peak / 1e3 + 10, r"$h_m F2$", color="w", fontsize=9)
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.015)
    cb.set_label(r"$\log_{10}\, n_e$ [m$^{-3}$]")
    fig.suptitle(
        "Burbuja de plasma ecuatorial (SI): capa F post-PRE ($h_mF2$ = 450 km), "
        "MUSCL + derivas implicitas + solve $\\sigma\\propto n$ + quimica",
        fontsize=11,
    )
    fig.savefig(os.path.join(OUT, "fig5_pluma_si.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"fig5_pluma_si.png en {OUT}")


if __name__ == "__main__":
    print(f"=== Pluma EPB SI: {NX}x{NZ} (dx = {DX / 1e3:.2f} km), "
          f"x: 0-{LX / 1e3:.0f} km, z: {Z0 / 1e3:.0f}-{Z1 / 1e3:.0f} km ===")
    print(f"    B = {B0 * 1e5:.2f}e-5 T, g = {G0:.2f} m/s^2, nu_in = {NU_IN}, "
          f"h_mF2 = {FR.h_peak / 1e3:.0f} km")
    plot(run())
