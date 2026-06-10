"""Cierre ionosferico realista para la rama ``EPB_TwoFluid``.

CAMINO CRITICO hacia resultados CONCRETOS de EPB (pasos 1-2):

    (1) Unidades fisicas SI y parametros de fondo de la region F ecuatorial.
    (2) Perfiles de fondo dependientes de la altura: densidad n_0(h) tipo
        Chapman, frecuencia de colision ion-neutro nu_in(h), tasa de
        recombinacion beta(h), campo magnetico dipolar B(h), gravedad g(h).
    (3) Termino quimico de produccion/perdida en CONTINUIDAD (lo que faltaba):
            dn/dt + div(n v) = P - beta n
        Sin recombinacion no hay umbral de disparo ni el gradiente afilado del
        bottomside que hace inestable la RT generalizada.

Este modulo es DELIBERADAMENTE SEPARADO del transporte (``epb_twofluid.py``) y
de las fuentes rigidas de momento / solve electrostatico (``epb_sources.py``),
siguiendo el principio de separacion del plan: transporte | fuentes de momento |
electrostatica | QUIMICA son piezas independientes.

--- Modelo continuo de la quimica (solo las filas de continuidad) ---

    dn_s/dt + div(n_s v_s) = P_s - beta_s n_s        (s = i, e)

con produccion fotoquimica P >= 0 y perdida lineal por recombinacion beta n.
En equilibrio fotoquimico n_0 = P / beta. La parte de transporte ya esta en
``epb_twofluid.transport_step``; aqui solo se trata el termino fuente local.

--- Discretizacion (backward Euler, A-estable y POSITIVA) ---

    n^{k+1} = (n^k + dt P) / (1 + dt beta)

Con P >= 0, beta >= 0 y n^k >= 0 el resultado es SIEMPRE >= 0 (preserva
positividad) y es incondicionalmente estable, lo cual es esencial porque beta a
baja altura es RIGIDA (recombinacion rapida). Es el integrador correcto para la
quimica acoplada al splitting IMEX.

--- Diagnostico fisico clave: tasa de crecimiento RT COLISIONAL ---

En la region F la RT generalizada NO esta en el limite inercial sqrt(g/L_n) sino
en el limite COLISIONAL dominado por nu_in (Sultan 1996, ignorando capa E y
vientos):

    gamma_RT = g / (nu_in L_n) - beta

La inestabilidad (gamma > 0) requiere g/(nu_in L_n) > beta, condicion que solo
se cumple por ENCIMA de cierta altura (donde nu_in y beta caen). Esto da el
UMBRAL DE ALTURA de aparicion de la EPB, un resultado concreto y comparable con
observaciones (tiempos de e-folding de ~minutos).
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
    """Parametros de referencia de la region F ecuatorial nocturna.

    Valores tipicos de un escenario de aparicion de EPB post-atardecer, en el que
    la capa F2 esta ELEVADA por el realce pre-inversion (PRE): el bottomside sube
    a alturas donde nu_in y la recombinacion son pequenas, condicion necesaria
    para que la RT colisional crezca (g/(nu_in L_n) > beta). Las alturas en
    metros. Los perfiles se construyen con ``build_background``.
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
    """Perfil de densidad de Chapman-alpha (capa F2).

    n_0(z) = n_max exp[ (1/2) (1 - u - exp(-u)) ],  u = (z - h_peak)/H.

    Tiene maximo en z = h_peak, decae rapido por encima y de forma cuasi-
    exponencial (escala H) por debajo: el bottomside afilado responsable de la
    RT generalizada.
    """
    u = (z - h_peak) / H
    return n_max * jnp.exp(0.5 * (1.0 - u - jnp.exp(-u)))


def exp_profile(z: jax.Array, v0: float, h0: float, H: float) -> jax.Array:
    """Perfil exponencial decreciente con la altura: v(z) = v0 exp(-(z-h0)/H).

    Usado para nu_in(h) y beta(h), que siguen la densidad neutra (decae con H
    propia de la termosfera).
    """
    return v0 * jnp.exp(-(z - h0) / H)


def dipole_B(z: jax.Array, B_surf_eq: float, RE: float = SI_CONST.RE) -> jax.Array:
    """Magnitud del campo dipolar ecuatorial: B(z) = B_surf_eq (RE/(RE+z))^3."""
    return B_surf_eq * (RE / (RE + z)) ** 3


def gravity(z: jax.Array, g_surf: float = 9.80665, RE: float = SI_CONST.RE) -> jax.Array:
    """Gravedad newtoniana: g(z) = g_surf (RE/(RE+z))^2."""
    return g_surf * (RE / (RE + z)) ** 2


class EPBBackground(NamedTuple):
    """Perfiles de fondo evaluados sobre una columna de alturas z (1D en z).

    Todos los arrays tienen la forma de ``z`` (perfil vertical). Para usarlos en
    el plano (z, x) basta con ``arr[:, None]`` (broadcast en x).
    """
    z: jax.Array          # alturas [m]
    n0: jax.Array         # densidad de equilibrio [m^-3]
    nu_in: jax.Array      # frecuencia de colision ion-neutro [s^-1]
    beta: jax.Array       # tasa de recombinacion [s^-1]
    prod: jax.Array       # produccion fotoquimica P = beta n0 [m^-3 s^-1]
    B: jax.Array          # |B| [T]
    g: jax.Array          # |g| [m/s^2]


def build_background(z: jax.Array, fr: FRegionParams = FRegionParams()) -> EPBBackground:
    """Construye todos los perfiles de fondo SI sobre la columna de alturas z.

    La produccion se fija AUTO-CONSISTENTE con el equilibrio fotoquimico,
    P(z) = beta(z) n_0(z), de modo que n_0 es un estado estacionario exacto de la
    quimica (la perturbacion se relaja hacia el).
    """
    n0 = chapman_layer(z, fr.n_max, fr.h_peak, fr.H_chapman)
    nu_in = exp_profile(z, fr.nu0, fr.h_ref, fr.H_nu)
    beta = exp_profile(z, fr.beta0, fr.h_ref, fr.H_beta)
    prod = beta * n0                       # equilibrio: P = beta n0
    B = dipole_B(z, fr.B_surf_eq)
    g = gravity(z, fr.g_surf)
    return EPBBackground(z=z, n0=n0, nu_in=nu_in, beta=beta, prod=prod, B=B, g=g)


def physparams_SI(fr: FRegionParams = FRegionParams()) -> EPBPhysParams:
    """Devuelve ``EPBPhysParams`` con constantes fisicas en SI (O+ de region F).

    Las velocidades del sonido resultantes son fisicas:
    c_i = sqrt(kB Ti / M_i) ~ 0.7 km/s,  c_e = sqrt(kB Te / M_e) ~ 120 km/s.
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
    """Avanza la quimica de continuidad un paso dt (backward Euler positivo).

    Resuelve  dn/dt = P - beta n  por celda:

        n^{k+1} = (n^k + dt P) / (1 + dt beta)

    A-estable, incondicionalmente positiva (P, beta, n >= 0 => n^{k+1} >= 0) y
    apta para beta rigida (recombinacion rapida a baja altura). ``prod`` y
    ``beta`` se broadcastean contra ``n`` (p.ej. perfiles verticales [:, None]).

    Las densidades de ambas especies se actualizan con la MISMA quimica
    (cuasineutralidad fotoquimica n_i ~ n_e); las corrientes no se tocan aqui.
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
    """Longitud de escala de densidad L_n = n / |dn/dz| (gradiente vertical).

    L_n caracteriza el bottomside: cuanto menor es L_n, mas inestable la RT.
    Se usa diferencia central; los bordes se replican.
    """
    dn = jnp.gradient(n0, dz)
    return n0 / jnp.maximum(jnp.abs(dn), DENS_FLOOR)


def collisional_growth_rate(
    g: jax.Array,
    nu_in: jax.Array,
    L_n: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    """Tasa de crecimiento RT COLISIONAL de la region F (Sultan 1996 reducido).

        gamma = g / (nu_in L_n) - beta

    Ignora el cortocircuito de la capa E (Sigma_E = 0) y los vientos neutros.
    gamma > 0 marca la condicion de inestabilidad (umbral de altura de la EPB).
    El tiempo de e-folding lineal es tau = 1/gamma.
    """
    return g / (nu_in * jnp.maximum(L_n, DENS_FLOOR)) - beta


def shielding_factor(sigma_F: jax.Array, sigma_E: jax.Array) -> jax.Array:
    """Factor de apantallamiento de la capa E (Paso 5):

        F_s = Sigma_F / (Sigma_F + Sigma_E)  en (0, 1].

    Las capas E conjugadas son una carga en PARALELO sobre el dinamo F: drenan
    la carga de polarizacion a lo largo de B y el campo E_p efectivo queda
    diluido por F_s. Con Sigma_E -> 0 (noche) F_s -> 1 (sin apantallamiento);
    con Sigma_E >> Sigma_F (dia) F_s -> 0 (RT suprimida).
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
    """Tasa RT colisional CON apantallamiento de capa E (Sultan 1996):

        gamma = [Sigma_F/(Sigma_F + Sigma_E)] g/(nu_in L_n) - beta

    Reduce a ``collisional_growth_rate`` cuando Sigma_E = 0. Es la forma que
    explica el control DIURNO de la EPB: de dia Sigma_E es grande y gamma < 0
    a toda altura; tras la puesta de sol la capa E recombina, F_s -> 1 y el
    umbral de altura reaparece.
    """
    Fs = shielding_factor(sigma_F, sigma_E)
    return Fs * g / (nu_in * jnp.maximum(L_n, DENS_FLOOR)) - beta
