# PyExner/solvers/kernels/epb_ionosphere.py
"""Cierre ionosferico en unidades SI para la rama EPB_TwoFluid.

Todo lo que convierte el modelo adimensional en una region F ecuatorial real:

  * constantes SI y parametros de fondo post-atardecer,
  * perfiles con la altura: n_0(h) tipo Chapman, nu_in(h), beta(h), B(h), g(h),
  * el termino quimico de continuidad, que era la pieza que faltaba:

        dn/dt + div(n v) = P - beta n

    Sin recombinacion no hay umbral de disparo ni el bottomside afilado que
    hace crecer la RT.

Mismo criterio de separacion que el resto de la rama: transporte, fuentes de
momento, electrostatica y quimica son piezas independientes; esta es la
cuarta. El transporte de div(n v) ya lo hace epb_twofluid; aca solo va el
termino local P - beta n.

Para la quimica uso backward Euler:

    n^{k+1} = (n^k + dt P) / (1 + dt beta)

A-estable y ademas positivo siempre (P, beta, n >= 0 => n^{k+1} >= 0), que es
lo que se necesita porque beta a baja altura es rigida.

El diagnostico central es la tasa RT colisional de la region F (Sultan 1996,
sin capa E ni vientos):

    gamma = g / (nu_in L_n) - beta

Ojo: NO es el limite inercial sqrt(g/L_n); en la region F manda nu_in. Que
gamma > 0 solo por encima de cierta altura es justamente el umbral de
aparicion de la EPB, y da e-foldings de minutos, comparables a observaciones.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_twofluid import EPBPhysParams, DENS_FLOOR


# --------------------------------------------------------------------------- #
# Constantes fisicas (SI)                                                      #
# --------------------------------------------------------------------------- #

class SI(NamedTuple):
    """Constantes fisicas fundamentales en unidades SI."""
    e: float = 1.602176634e-19        # carga elemental [C]
    kB: float = 1.380649e-23          # constante de Boltzmann [J/K]
    amu: float = 1.66053906660e-27    # unidad de masa atomica [kg]
    Me: float = 9.1093837015e-31      # masa del electron [kg]
    RE: float = 6.371e6               # radio terrestre [m]


SI_CONST = SI()

# Ion dominante de la region F: oxigeno atomico ionizado O+.
M_OPLUS = 16.0 * SI_CONST.amu         # masa del O+ [kg]


# --------------------------------------------------------------------------- #
# Parametros de fondo de la region F ecuatorial (valores de referencia)       #
# --------------------------------------------------------------------------- #

class FRegionParams(NamedTuple):
    """Region F ecuatorial nocturna de referencia.

    Escenario tipico de aparicion de EPB: la capa F2 elevada por el PRE, con
    el bottomside a alturas donde nu_in y beta ya son chicas (condicion para
    que g/(nu_in L_n) > beta). Alturas en metros. Los perfiles completos los
    arma ``build_background``.
    """
    # Quimica / densidad
    n_max: float = 1.0e12             # densidad de pico de la capa F2 [m^-3]
    h_peak: float = 400.0e3           # altura del pico h_mF2 ELEVADO (PRE) [m]
    H_chapman: float = 50.0e3         # escala de la capa Chapman [m]
    # Recombinacion (perdida lineal): beta(h) = beta0 exp(-(h-h0)/H_beta)
    beta0: float = 2.0e-4             # tasa de recombinacion en h_ref [s^-1]
    h_ref: float = 300.0e3            # altura de referencia [m]
    H_beta: float = 30.0e3            # escala de caida de la recombinacion [m]
    # Colisiones ion-neutro: nu_in(h) = nu0 exp(-(h-h_ref)/H_nu)
    nu0: float = 0.5                  # nu_in en h_ref [s^-1]
    H_nu: float = 40.0e3              # escala de caida de la colision [m]
    # Temperaturas (isotermo)
    Ti: float = 1000.0                # temperatura ionica [K]
    Te: float = 1000.0                # temperatura electronica [K]
    # Campo magnetico ecuatorial en superficie (dipolo)
    B_surf_eq: float = 3.12e-5        # |B| ecuatorial en superficie [T]
    # Gravedad de superficie
    g_surf: float = 9.80665           # [m/s^2]


# --------------------------------------------------------------------------- #
# Perfiles de fondo dependientes de la altura                                 #
# --------------------------------------------------------------------------- #

def chapman_layer(z: jax.Array, n_max: float, h_peak: float, H: float) -> jax.Array:
    """Capa de Chapman-alpha:

    n_0(z) = n_max exp[ (1/2)(1 - u - e^{-u}) ],  u = (z - h_peak)/H.

    Pico en h_peak, caida suave arriba y cuasi-exponencial (escala H) abajo:
    ese bottomside afilado es el que desestabiliza la RT.
    """
    u = (z - h_peak) / H
    return n_max * jnp.exp(0.5 * (1.0 - u - jnp.exp(-u)))


def exp_profile(z: jax.Array, v0: float, h0: float, H: float) -> jax.Array:
    """Exponencial decreciente v(z) = v0 exp(-(z-h0)/H).

    Lo uso para nu_in(h) y beta(h): ambas siguen a la densidad neutra.
    """
    return v0 * jnp.exp(-(z - h0) / H)


def dipole_B(z: jax.Array, B_surf_eq: float, RE: float = SI_CONST.RE) -> jax.Array:
    """Magnitud del campo dipolar ecuatorial: B(z) = B_surf_eq (RE/(RE+z))^3."""
    return B_surf_eq * (RE / (RE + z)) ** 3


def gravity(z: jax.Array, g_surf: float = 9.80665, RE: float = SI_CONST.RE) -> jax.Array:
    """Gravedad newtoniana: g(z) = g_surf (RE/(RE+z))^2."""
    return g_surf * (RE / (RE + z)) ** 2


class EPBBackground(NamedTuple):
    """Perfiles de fondo sobre una columna de alturas z (todo 1D en z).

    Para usarlos en el plano (z, x) alcanza con arr[:, None] (broadcast en x).
    """
    z: jax.Array          # alturas [m]
    n0: jax.Array         # densidad de equilibrio [m^-3]
    nu_in: jax.Array      # frecuencia de colision ion-neutro [s^-1]
    beta: jax.Array       # tasa de recombinacion [s^-1]
    prod: jax.Array       # produccion fotoquimica P = beta n0 [m^-3 s^-1]
    B: jax.Array          # |B| [T]
    g: jax.Array          # |g| [m/s^2]


def build_background(z: jax.Array, fr: FRegionParams = FRegionParams()) -> EPBBackground:
    """Arma todos los perfiles de fondo SI sobre la columna z.

    La produccion se fija en equilibrio fotoquimico, P = beta n_0, asi n_0 es
    estado estacionario exacto de la quimica y cualquier perturbacion relaja
    hacia el.
    """
    n0 = chapman_layer(z, fr.n_max, fr.h_peak, fr.H_chapman)
    nu_in = exp_profile(z, fr.nu0, fr.h_ref, fr.H_nu)
    beta = exp_profile(z, fr.beta0, fr.h_ref, fr.H_beta)
    prod = beta * n0                       # equilibrio: P = beta n0
    B = dipole_B(z, fr.B_surf_eq)
    g = gravity(z, fr.g_surf)
    return EPBBackground(z=z, n0=n0, nu_in=nu_in, beta=beta, prod=prod, B=B, g=g)


def physparams_SI(fr: FRegionParams = FRegionParams()) -> EPBPhysParams:
    """EPBPhysParams con constantes SI (O+ de region F).

    Da velocidades del sonido fisicas: c_i ~ 0.7 km/s, c_e ~ 120 km/s. Ese
    c_e es el que domina el CFL cuando se corre en SI.
    """
    return EPBPhysParams(
        e=SI_CONST.e,
        kB=SI_CONST.kB,
        Mi=M_OPLUS,
        Me=SI_CONST.Me,
        Ti=fr.Ti,
        Te=fr.Te,
    )


# --------------------------------------------------------------------------- #
# Operador quimico: produccion - recombinacion (backward Euler positivo)       #
# --------------------------------------------------------------------------- #

def chemistry_implicit_step(
    n_i: jax.Array,
    n_e: jax.Array,
    dt: float,
    prod: jax.Array,
    beta: jax.Array,
) -> tuple:
    """Un paso dt de quimica, backward Euler.

    Resuelve dn/dt = P - beta n celda a celda:

        n^{k+1} = (n^k + dt P) / (1 + dt beta)

    Estable para cualquier dt y positivo siempre que P, beta, n >= 0; con eso
    aguanta la beta rigida de baja altura sin drama. prod y beta se
    broadcastean contra n (perfiles verticales van como [:, None]).

    Las dos especies llevan la misma quimica (cuasineutralidad fotoquimica);
    las corrientes no se tocan aca.
    """
    denom = 1.0 + dt * beta
    n_i_new = (n_i + dt * prod) / denom
    n_e_new = (n_e + dt * prod) / denom
    return n_i_new, n_e_new


def chemistry_step_state(state, dt: float, prod: jax.Array, beta: jax.Array):
    """Igual que ``chemistry_implicit_step`` pero sobre un ``EPBTwoFluidState``.

    Devuelve un estado nuevo con n_i, n_e actualizados y el resto intacto.
    """
    n_i_new, n_e_new = chemistry_implicit_step(state.n_i, state.n_e, dt, prod, beta)
    return state.replace(n_i=n_i_new, n_e=n_e_new)


# --------------------------------------------------------------------------- #
# Diagnosticos fisicos                                                          #
# --------------------------------------------------------------------------- #

def density_scale_length(n0: jax.Array, dz: float) -> jax.Array:
    """L_n = n / |dn/dz|. Cuanto mas corto el L_n del bottomside, mas
    inestable la RT. Diferencia central, bordes replicados."""
    dn = jnp.gradient(n0, dz)
    return n0 / jnp.maximum(jnp.abs(dn), DENS_FLOOR)


def collisional_growth_rate(
    g: jax.Array,
    nu_in: jax.Array,
    L_n: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    """Tasa RT colisional de la region F (Sultan 1996, forma reducida):

        gamma = g / (nu_in L_n) - beta

    Sin capa E (Sigma_E = 0) ni vientos. gamma > 0 marca la inestabilidad; el
    e-folding lineal es tau = 1/gamma.
    """
    return g / (nu_in * jnp.maximum(L_n, DENS_FLOOR)) - beta


def shielding_factor(sigma_F: jax.Array, sigma_E: jax.Array) -> jax.Array:
    """Apantallamiento de la capa E: F_s = Sigma_F/(Sigma_F + Sigma_E) en (0,1].

    Las capas E conjugadas son una carga en paralelo sobre el dinamo F: drenan
    la polarizacion a lo largo de B y el campo queda diluido por F_s. De noche
    Sigma_E -> 0 y F_s -> 1; de dia Sigma_E >> Sigma_F y la RT se apaga.
    """
    return sigma_F / jnp.maximum(sigma_F + sigma_E, DENS_FLOOR)


def collisional_growth_rate_shielded(
    g: jax.Array,
    nu_in: jax.Array,
    L_n: jax.Array,
    beta: jax.Array,
    sigma_F: jax.Array,
    sigma_E: jax.Array,
) -> jax.Array:
    """Tasa RT colisional con apantallamiento de capa E (Sultan 1996):

        gamma = [Sigma_F/(Sigma_F + Sigma_E)] g/(nu_in L_n) - beta

    Con Sigma_E = 0 reduce a collisional_growth_rate. Esta es la forma que
    explica por que la EPB es nocturna: de dia Sigma_E es enorme y gamma < 0
    a toda altura; al recombinar la capa E tras la puesta de sol, F_s -> 1 y
    reaparece el umbral.
    """
    Fs = shielding_factor(sigma_F, sigma_E)
    return Fs * g / (nu_in * jnp.maximum(L_n, DENS_FLOOR)) - beta
