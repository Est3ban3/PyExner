# PyExner/solvers/kernels/epb_fluxtube.py
"""Geometria dipolar integrada en tubo de flujo.

La dinamica perpendicular de la EPB es 2D en el plano ecuatorial porque la
conductividad paralela es enorme: cada linea de B es equipotencial, asi que
cada celda del plano representa un tubo de flujo completo. Los coeficientes
que alimentan al modelo 2D no son valores locales del apex sino integrales a
lo largo de la linea dipolar. Este modulo construye ese mapeo:

    h_apex -> (Sigma_P^F, Sigma_P^E, N_FT, nu_eff, F_s, gamma_FT)

Geometria (coordenada: latitud magnetica lambda), linea con apex h_apex:

    L = (RE + h_apex)/RE
    r(lambda)  = L RE cos^2(lambda)
    ds         = L RE cos(lambda) sqrt(1 + 3 sin^2 lambda) dlambda
    B(lambda)  = (B_eq/L^3) sqrt(1 + 3 sin^2 lambda) / cos^6(lambda)

La linea se corta en h_min ~ 90 km (base de la capa E); por debajo ya no hay
plasma que cuente.

Conductividad Pedersen (formula completa, vale en E y F):

    sigma_P = (n e/B) nu Omega_i / (nu^2 + Omega_i^2),  Omega_i = e B/M_i

En la F (Omega >> nu) es chica, ~ n nu; en la E (nu ~ Omega) es maxima: ese
es el cortocircuito diurno.

Conductancias y promedios (el factor 2 es por los dos hemisferios):

    Sigma_P^{F,E} = 2 int sigma_P ds
    N_FT   = 2 int n_F ds
    nu_eff = 2 int n_F nu ds / N_FT
    F_s    = Sigma_F/(Sigma_F + Sigma_E)

y la tasa RT de tubo de flujo (Sultan 1996, sin vientos ni deriva V_P):

    gamma_FT(h_apex) = F_s g/(nu_eff L_n) - beta

con L_n del bottomside Chapman analitico: L_n = 2H/(e^{-u} - 1), que solo es
positivo bajo el pico (el topside es RT-estable).

Nota de implementacion: esto es host-side (NumPy). Son precomputos lentos que
alimentan a los kernels JAX (p.ej. el shunt R_E del solve con sigma); no van
dentro del lazo jiteado. Los perfiles se reusan de epb_ionosphere para tener
una sola fuente de verdad de las formas.

Simplificaciones asumidas (conscientes): ion unico O+ tambien en la capa E
(los reales alli son NO+/O2+, factor ~2 en Omega_i); nu_in con dos
exponenciales (termosfera F + base E) calibradas a nu(105 km) ~ 3e3 s^-1 y
nu(300 km) ~ 0.5 s^-1; sin vientos neutros; el PRE entra por epb_pre.
"""

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_ionosphere import (
    SI_CONST, M_OPLUS, FRegionParams,
    chapman_layer, exp_profile, gravity,
)


# --------------------------------------------------------------------------- #
# Parametros de la capa E                                                      #
# --------------------------------------------------------------------------- #

class ELayerParams(NamedTuple):
    """Capa E ecuatorial (Chapman delgada) + colision de su base.

    Por defecto nocturna: tras la puesta de sol la capa E recombina rapido
    (recombinacion disociativa de NO+/O2+) y queda un remanente debil. Para
    el caso diurno esta ELAYER_DAY.
    """
    n_max_E: float = 5.0e9      # densidad de pico de la capa E nocturna [m^-3]
    h_peak_E: float = 105.0e3   # altura del pico de la capa E [m]
    H_E: float = 6.0e3          # escala de la capa E [m]
    # Colision ion-neutro de la baja termosfera (domina cerca de la capa E):
    nuE0: float = 3.0e3         # nu_in en h_peak_E [s^-1]
    H_nuE: float = 7.0e3        # escala de caida [m]


ELAYER_NIGHT = ELayerParams()
ELAYER_DAY = ELayerParams(n_max_E=1.5e11)   # capa E diurna (fotoionizada)

H_MIN = 90.0e3                  # base de la ionosfera: truncado de la linea [m]


def nu_in_total(h, fr: FRegionParams, el: ELayerParams):
    """nu_in(h) con dos exponenciales: termosfera F + base de la capa E.

    El termino F (escala 40 km) da nu ~ 0.5 s^-1 a 300 km; el termino E
    (escala 7 km) da nu ~ 3e3 s^-1 a 105 km y muere por encima de ~150 km.
    Con una sola exponencial extrapolada a la baja termosfera nu quedaba
    ordenes de magnitud corta.
    """
    nu_F = np.asarray(exp_profile(jnp.asarray(h), fr.nu0, fr.h_ref, fr.H_nu))
    nu_E = np.asarray(exp_profile(jnp.asarray(h), el.nuE0, el.h_peak_E, el.H_nuE))
    return nu_F + nu_E


def pedersen_conductivity(n, nu, B, M: float = M_OPLUS):
    """sigma_P = (n e/B) nu Omega/(nu^2 + Omega^2), Omega = e B/M  [S/m]."""
    e = SI_CONST.e
    Om = e * B / M
    return (n * e / B) * nu * Om / (nu * nu + Om * Om)


# --------------------------------------------------------------------------- #
# Geometria de la linea dipolar                                                #
# --------------------------------------------------------------------------- #

class DipoleLine(NamedTuple):
    """Discretizacion de media linea dipolar (lambda en [0, lambda_max])."""
    lam: np.ndarray       # latitud magnetica [rad], (A, nlat)
    h: np.ndarray         # altura [m], (A, nlat)
    B: np.ndarray         # |B| [T], (A, nlat)
    dsdlam: np.ndarray    # ds/dlambda [m/rad], (A, nlat)


def dipole_line(h_apex, fr: FRegionParams = FRegionParams(),
                nlat: int = 801, h_min: float = H_MIN) -> DipoleLine:
    """Construye la(s) linea(s) dipolar(es) con apex ``h_apex`` (escalar o (A,)).

    La malla en lambda va de 0 (apex) a lambda_max (pie a h = h_min):
        cos^2(lambda_max) = (RE + h_min) / (L RE).
    """
    RE = SI_CONST.RE
    h_apex = np.atleast_1d(np.asarray(h_apex, dtype=float))      # (A,)
    L = (RE + h_apex) / RE                                       # (A,)
    lam_max = np.arccos(np.sqrt((RE + h_min) / (L * RE)))        # (A,)
    s = np.linspace(0.0, 1.0, nlat)                              # (nlat,)
    lam = lam_max[:, None] * s[None, :]                          # (A, nlat)
    cl = np.cos(lam)
    s3 = np.sqrt(1.0 + 3.0 * np.sin(lam) ** 2)
    r = L[:, None] * RE * cl * cl
    h = r - RE
    B = (fr.B_surf_eq / L[:, None] ** 3) * s3 / cl**6
    dsdlam = L[:, None] * RE * cl * s3
    return DipoleLine(lam=lam, h=h, B=B, dsdlam=dsdlam)


# --------------------------------------------------------------------------- #
# Cantidades integradas en el tubo de flujo                                    #
# --------------------------------------------------------------------------- #

class FluxTubeQuantities(NamedTuple):
    """Cantidades integradas, todas de forma (A,) (una por apex)."""
    h_apex: np.ndarray    # altura del apex [m]
    Sigma_F: np.ndarray   # conductancia Pedersen del plasma F [S]
    Sigma_E: np.ndarray   # conductancia Pedersen del plasma E [S]
    N_FT: np.ndarray      # contenido del tubo int n_F ds [m^-2]
    nu_eff: np.ndarray    # colision ion-neutro ponderada por n_F [s^-1]
    F_s: np.ndarray       # factor de apantallamiento Sigma_F/(Sigma_F+Sigma_E)


def flux_tube_quantities(h_apex, fr: FRegionParams = FRegionParams(),
                         el: ELayerParams = ELAYER_NIGHT,
                         nlat: int = 801) -> FluxTubeQuantities:
    """Integra perfiles a lo largo de la linea dipolar (trapecio, x2 hemisferios)."""
    line = dipole_line(h_apex, fr, nlat)
    h, B, lam, w = line.h, line.B, line.lam, line.dsdlam

    n_F = np.asarray(chapman_layer(jnp.asarray(h), fr.n_max, fr.h_peak, fr.H_chapman))
    n_E = np.asarray(chapman_layer(jnp.asarray(h), el.n_max_E, el.h_peak_E, el.H_E))
    nu = nu_in_total(h, fr, el)

    sig_F = pedersen_conductivity(n_F, nu, B)
    sig_E = pedersen_conductivity(n_E, nu, B)

    def _int(f):    # 2 x integral sobre media linea (simetria hemisferica)
        return 2.0 * np.trapezoid(f * w, lam, axis=1)

    Sigma_F = _int(sig_F)
    Sigma_E = _int(sig_E)
    N_FT = _int(n_F)
    nu_eff = _int(n_F * nu) / np.maximum(N_FT, 1e-300)
    F_s = Sigma_F / np.maximum(Sigma_F + Sigma_E, 1e-300)
    return FluxTubeQuantities(
        h_apex=np.atleast_1d(np.asarray(h_apex, dtype=float)),
        Sigma_F=Sigma_F, Sigma_E=Sigma_E, N_FT=N_FT, nu_eff=nu_eff, F_s=F_s,
    )


# --------------------------------------------------------------------------- #
# Tasa de crecimiento RT flux-tube                                             #
# --------------------------------------------------------------------------- #

def bottomside_Ln(h, fr: FRegionParams = FRegionParams()):
    """L_n analitico del bottomside Chapman: L_n = 2H/(e^{-u} - 1), u=(h-hp)/H.

    Positivo solo bajo el pico (gradiente hacia arriba con g hacia abajo =
    RT-inestable). Sobre el pico devuelve +inf, o sea 1/L_n = 0: topside
    estable.
    """
    h = np.asarray(h, dtype=float)
    u = (h - fr.h_peak) / fr.H_chapman
    denom = np.expm1(-u)                       # e^{-u} - 1
    with np.errstate(divide="ignore"):
        Ln = np.where(denom > 0.0, 2.0 * fr.H_chapman / np.maximum(denom, 1e-300),
                      np.inf)
    return Ln


def gamma_flux_tube(h_apex, fr: FRegionParams = FRegionParams(),
                    el: ELayerParams = ELAYER_NIGHT, nlat: int = 801):
    """gamma_FT(h_apex) = F_s g/(nu_eff L_n) - beta (Sultan reducido, FT).

    Devuelve (gamma, ft), gamma de forma (A,) y ft las cantidades integradas
    por si hace falta diagnostico. g y beta van evaluadas en el apex; nu_eff
    y F_s son del tubo completo.
    """
    ft = flux_tube_quantities(h_apex, fr, el, nlat)
    g_a = np.asarray(gravity(jnp.asarray(ft.h_apex)))
    beta_a = np.asarray(exp_profile(jnp.asarray(ft.h_apex), fr.beta0, fr.h_ref, fr.H_beta))
    Ln = bottomside_Ln(ft.h_apex, fr)
    drive = np.where(np.isfinite(Ln), ft.F_s * g_a / (ft.nu_eff * Ln), 0.0)
    return drive - beta_a, ft


def gamma_bottomside_max(fr: FRegionParams = FRegionParams(),
                         el: ELayerParams = ELAYER_NIGHT,
                         n_apex: int = 9, span=(0.5, 3.0), nlat: int = 401):
    """Maximo de gamma_FT barriendo apexes del bottomside [hp-s1 H, hp-s0 H].

    Bajando el gradiente relativo mejora pero nu_eff sube, asi que el maximo
    lo decide el balance: se barre y se toma el mejor. Devuelve
    (gamma_max, h_apex_max). Es el gamma "del perfil" que consume el
    disparador PRE.
    """
    s0, s1 = span
    h_apexes = fr.h_peak - np.linspace(s0, s1, n_apex) * fr.H_chapman
    h_apexes = np.maximum(h_apexes, H_MIN + 30.0e3)
    gam, _ = gamma_flux_tube(h_apexes, fr, el, nlat)
    k = int(np.argmax(gam))
    return float(gam[k]), float(h_apexes[k])
