"""Figuras tipicas de EPB a partir de los kernels validados (Pasos 1-7).

Genera en ``tests/runtime/EPB/figures/``:

  fig1_perfiles.png    Perfiles de fondo vs altura: n0 Chapman, nu_in, beta y
                       gamma local (umbral de altura) — el "ionograma" clasico.
  fig2_fluxtube.png    Conductancias integradas Sigma_F / Sigma_E, factor de
                       apantallamiento F_s y gamma_FT(h_apex) noche vs dia —
                       la curva de crecimiento vs altura del apex (Sultan).
  fig3_pre_onset.png   Tarde-noche: drift V(t), subida de la capa h(t), decaida
                       de la capa E, gamma(t) y e-folds acumulados con la hora
                       de onset — el diagrama de disparo por PRE.
  fig4_morfologia.png  Mapas n(x,z) de la corrida RT COLISIONAL no lineal
                       (MUSCL + fuentes implicitas con nu_in > 0: la caida
                       libre por polarizacion queda frenada a g/nu_in y la
                       cara inferior desarrolla los dedos/burbujas clasicos).

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 JAX_PLATFORMS=cpu python plot_epb.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_ionosphere import (
    FRegionParams, build_background, density_scale_length,
    collisional_growth_rate,
)
from PyExner.solvers.kernels.epb_fluxtube import (
    ELAYER_DAY, ELAYER_NIGHT, flux_tube_quantities, gamma_flux_tube,
)
from PyExner.solvers.kernels.epb_pre import (
    GROWTH_THRESHOLD, PREParams, onset_prediction,
)
from PyExner.solvers.kernels.epb_twofluid import (
    EPBPhysParams, stack_state, unstack_state, transport_step_muscl,
)
from PyExner.solvers.kernels.epb_sources import (
    EPBSourceParams, implicit_source_solve,
)
from PyExner.state.epb_twofluid_state import EPBTwoFluidState

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

FR = FRegionParams()


# --------------------------------------------------------------------------- #
# Figura 1: perfiles de fondo vs altura                                        #
# --------------------------------------------------------------------------- #

def fig1_perfiles():
    z = jnp.linspace(200.0e3, 600.0e3, 600)
    bg = build_background(z, FR)
    h_km = np.asarray(z) / 1e3
    Ln = density_scale_length(bg.n0, float(z[1] - z[0]))
    gam = np.asarray(collisional_growth_rate(bg.g, bg.nu_in, Ln, bg.beta))
    # Mascara fisica: sin plasma (n0 << n_max) L_n es ruido numerico y gamma
    # carece de sentido; el diagnostico solo aplica donde hay capa.
    n0 = np.asarray(bg.n0)
    gam = np.where(n0 > 1e-4 * FR.n_max, gam, np.nan)
    # Signo del gradiente: solo el BOTTOMSIDE (dn/dz > 0, denso encima de
    # ligero con g hacia abajo) es RT-inestable; en el topside el drive es
    # estabilizante => gamma = -beta (igual criterio que bottomside_Ln, Paso 6).
    dndz = np.gradient(n0, float(z[1] - z[0]))
    gam = np.where(dndz > 0.0, gam, -np.asarray(bg.beta))

    fig, axs = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)

    axs[0].semilogx(np.asarray(bg.n0), h_km, "C0", lw=2)
    axs[0].axhline(FR.h_peak / 1e3, color="gray", ls=":", lw=1)
    axs[0].text(2e10, FR.h_peak / 1e3 + 6, r"$h_{m}F2$", color="gray")
    axs[0].set_xlim(1e6, 3e12)
    axs[0].set_xlabel(r"$n_0$ [m$^{-3}$]")
    axs[0].set_ylabel("altura [km]")
    axs[0].set_title("capa F2 (Chapman)")

    axs[1].semilogx(np.asarray(bg.nu_in), h_km, "C1", lw=2)
    axs[1].set_xlabel(r"$\nu_{in}$ [s$^{-1}$]")
    axs[1].set_title("colision ion-neutro")

    axs[2].semilogx(np.asarray(bg.beta), h_km, "C2", lw=2)
    axs[2].set_xlabel(r"$\beta$ [s$^{-1}$]")
    axs[2].set_title("recombinacion")

    axs[3].plot(gam * 1e3, h_km, "C3", lw=2)
    axs[3].axvline(0.0, color="k", lw=0.8)
    valid = np.isfinite(gam)
    pos = np.where(valid & (gam > 0))[0]
    if len(pos):
        axs[3].axhline(h_km[pos[0]], color="C3", ls="--", lw=1)
        axs[3].text(0.05, h_km[pos[0]] + 8,
                    f"umbral {h_km[pos[0]]:.0f} km", color="C3")
    gmax = np.nanmax(np.abs(gam))
    axs[3].set_xlim(-1.1 * gmax * 1e3, 1.1 * gmax * 1e3)
    axs[3].set_xlabel(r"$\gamma = g/(\nu_{in} L_n) - \beta$  [$10^{-3}$ s$^{-1}$]")
    axs[3].set_title("RT colisional local")

    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle("Fondo ionosferico ecuatorial post-atardecer (Pasos 1-2)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_perfiles.png"), dpi=150)
    plt.close(fig)
    print("fig1_perfiles.png")


# --------------------------------------------------------------------------- #
# Figura 2: flux-tube — conductancias y gamma_FT(h_apex)                       #
# --------------------------------------------------------------------------- #

def fig2_fluxtube():
    h_apex = np.linspace(220.0e3, 550.0e3, 60)
    ftN = flux_tube_quantities(h_apex, FR, ELAYER_NIGHT)
    ftD = flux_tube_quantities(h_apex, FR, ELAYER_DAY)
    gamN, _ = gamma_flux_tube(h_apex, FR, ELAYER_NIGHT)
    gamD, _ = gamma_flux_tube(h_apex, FR, ELAYER_DAY)
    hk = h_apex / 1e3

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))

    axs[0].semilogy(hk, ftN.Sigma_F, "C0", lw=2, label=r"$\Sigma_P^F$")
    axs[0].semilogy(hk, ftN.Sigma_E, "C1", lw=2, label=r"$\Sigma_P^E$ noche")
    axs[0].semilogy(hk, ftD.Sigma_E, "C1", lw=2, ls="--", label=r"$\Sigma_P^E$ dia")
    axs[0].set_xlabel(r"$h_{apex}$ [km]")
    axs[0].set_ylabel("conductancia [S]")
    axs[0].set_title("conductancias integradas en el tubo")
    axs[0].legend()

    axs[1].plot(hk, ftN.F_s, "C2", lw=2, label="noche")
    axs[1].plot(hk, ftD.F_s, "C2", lw=2, ls="--", label="dia")
    axs[1].set_xlabel(r"$h_{apex}$ [km]")
    axs[1].set_ylabel(r"$F_s = \Sigma_F/(\Sigma_F+\Sigma_E)$")
    axs[1].set_ylim(0, 1.02)
    axs[1].set_title("factor de apantallamiento")
    axs[1].legend()

    axs[2].plot(hk, gamN * 1e3, "C3", lw=2, label="noche")
    axs[2].plot(hk, gamD * 1e3, "C3", lw=2, ls="--", label="dia")
    axs[2].axhline(0.0, color="k", lw=0.8)
    pos = np.where(gamN > 0)[0]
    if len(pos):
        axs[2].axvline(hk[pos[0]], color="C3", ls=":", lw=1)
        axs[2].text(hk[pos[0]] + 5, -0.35, f"umbral\n{hk[pos[0]]:.0f} km",
                    color="C3", fontsize=9)
    k = int(np.argmax(gamN))
    axs[2].annotate(
        rf"$\tau_{{min}}$ = {1.0 / gamN[k] / 60.0:.0f} min",
        xy=(hk[k], gamN[k] * 1e3), xytext=(hk[k] + 40, gamN[k] * 1e3),
        arrowprops=dict(arrowstyle="->", color="k"), fontsize=9,
    )
    axs[2].set_xlabel(r"$h_{apex}$ [km]")
    axs[2].set_ylabel(r"$\gamma_{FT}$ [$10^{-3}$ s$^{-1}$]")
    axs[2].set_title("RT flux-tube (Sultan)")
    axs[2].legend()

    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle("Cantidades integradas en tubo de flujo dipolar (Pasos 5-6)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_fluxtube.png"), dpi=150)
    plt.close(fig)
    print("fig2_fluxtube.png")


# --------------------------------------------------------------------------- #
# Figura 3: tarde-noche del PRE y onset                                        #
# --------------------------------------------------------------------------- #

def fig3_pre_onset():
    p = PREParams()
    cast = onset_prediction(p)
    cast0 = onset_prediction(p._replace(V_pre=0.0))
    t = cast.t_lt

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(t, cast.V, "C0", lw=2, label="con PRE")
    axs[0].plot(cast0.t_lt, cast0.V, "C0", ls="--", lw=1.5, label="sin PRE")
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_ylabel("drift V [m/s]")
    axs[0].set_title("disparador PRE y onset de la EPB (Paso 7)")
    axs[0].legend(loc="upper left")

    axs[1].plot(t, cast.h_peak / 1e3, "C1", lw=2)
    axs[1].plot(cast0.t_lt, cast0.h_peak / 1e3, "C1", ls="--", lw=1.5)
    axs[1].set_ylabel(r"$h_m F2$ [km]")
    ax1b = axs[1].twinx()
    ax1b.semilogy(t, cast.n_E, "C4", lw=1.5)
    ax1b.set_ylabel(r"$n_{max}^E$ [m$^{-3}$]", color="C4")
    ax1b.tick_params(axis="y", colors="C4")

    axs[2].plot(t, cast.gamma * 1e3, "C3", lw=2)
    axs[2].plot(cast0.t_lt, cast0.gamma * 1e3, "C3", ls="--", lw=1.5)
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].set_ylabel(r"$\gamma_{FT}$ [$10^{-3}$ s$^{-1}$]")

    axs[3].plot(t, cast.Gamma, "C2", lw=2, label="con PRE")
    axs[3].plot(cast0.t_lt, cast0.Gamma, "C2", ls="--", lw=1.5, label="sin PRE")
    axs[3].axhline(GROWTH_THRESHOLD, color="gray", ls=":",
                   label=r"$\ln 10^3 \approx 6.9$")
    if cast.t_onset is not None:
        for ax in axs:
            ax.axvline(cast.t_onset, color="m", ls="-.", lw=1.2)
        axs[3].text(cast.t_onset + 0.1, 1.0,
                    f"onset {cast.t_onset:.2f} LT", color="m", rotation=90)
    axs[3].set_ylabel(r"$\Gamma=\int\gamma_+\,dt$ [e-folds]")
    axs[3].set_xlabel("hora local [h]")
    axs[3].legend(loc="upper left")

    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_pre_onset.png"), dpi=150)
    plt.close(fig)
    print("fig3_pre_onset.png")


# --------------------------------------------------------------------------- #
# Figura 4: morfologia 2D de la burbuja (RT no lineal, MUSCL + fuentes)        #
# --------------------------------------------------------------------------- #

PHYS = EPBPhysParams(Ti=0.0, Te=0.0)
LX = LZ = 4.0
N = 96
DX = LX / N
G_RT, B_RT = 1.0, 1.0
NU_IN = 0.5                       # colisional moderado: v_term = g nu/(nu^2+W^2)
                                  # = 0.4 y g_eff = g W^2/(nu^2+W^2) = 0.8 (W=1)
N_LOW, N_HIGH = 0.2, 1.0
Z_C, WIDTH, EPS, KMODE = LZ / 2.0, 0.3, 2e-2, 3


def fig4_morfologia():
    xc = (np.arange(N) + 0.5) * DX
    zc = (np.arange(N) + 0.5) * DX
    X, Z = np.meshgrid(xc, zc, indexing="xy")
    base = N_LOW + (N_HIGH - N_LOW) * np.exp(-((Z - Z_C) ** 2) / (2.0 * WIDTH**2))
    env = np.exp(-((Z - (Z_C - WIDTH)) ** 2) / (2.0 * WIDTH**2))
    n = base * (1.0 + EPS * np.cos(2 * np.pi * KMODE * X / LX) * env)
    zeros = jnp.zeros((N, N))
    state = EPBTwoFluidState(n_i=jnp.asarray(n), n_e=jnp.asarray(n),
                             j_ix=zeros, j_iy=zeros, j_iz=zeros,
                             j_ex=zeros, j_ey=zeros, j_ez=zeros)

    src = EPBSourceParams(gz=-G_RT, By=B_RT, nu_in=NU_IN, nu_en=NU_IN)
    v_char = max(PHYS.Mi, PHYS.Me) * G_RT / (PHYS.e * B_RT)
    dt = 0.3 * DX / max(v_char, 1e-3)

    @jax.jit
    def step(s):
        s = transport_step_muscl(s, dt, DX, PHYS)
        return implicit_source_solve(s, dt, PHYS, src, DX)

    n_snap = 4
    steps_total = 360                 # t_final = 4.5 (fase no lineal)
    every = steps_total // (n_snap - 1)

    def _center(nk):
        """Marco co-movil: roll periodico en z para centrar la banda en Z_C.
        Exacto bajo BC periodicas (cambio de marco galileano en z)."""
        prof = nk.mean(axis=1)
        k_band = int(np.argmax(prof))
        k_target = int(Z_C / DX)
        return np.roll(nk, k_target - k_band, axis=0)

    snaps = [(0.0, _center(np.asarray(state.n_i)))]
    for k in range(1, steps_total + 1):
        state = step(state)
        if k % every == 0:
            snaps.append((k * dt, _center(np.asarray(state.n_i))))

    fig, axs = plt.subplots(1, len(snaps), figsize=(3.4 * len(snaps), 3.9),
                            sharey=True)
    vmin, vmax = N_LOW * 0.5, N_HIGH * 1.05
    for ax, (tk, nk) in zip(axs, snaps):
        im = ax.pcolormesh(xc, zc, nk, cmap="viridis", vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.set_title(f"t = {tk:.1f}")
        ax.set_xlabel("x")
        ax.set_aspect("equal")
    axs[0].set_ylabel("z (altura, marco co-movil)")
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.015)
    cb.set_label(r"$n_i$ (adim.)")
    fig.suptitle("Burbuja RT colisional: depleccion ascendiendo por el "
                 "bottomside (MUSCL + fuentes implicitas, $g=-g\\hat z$, "
                 f"$B=B\\hat y$, $\\nu_{{in}}={NU_IN:.1f}$)")
    fig.savefig(os.path.join(OUT, "fig4_morfologia.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("fig4_morfologia.png")


if __name__ == "__main__":
    fig1_perfiles()
    fig2_fluxtube()
    fig3_pre_onset()
    fig4_morfologia()
    print(f"\nFiguras en: {OUT}")
