# PyExner/solvers/kernels/epb_pre.py
"""Disparador PRE: campo de fondo E_0(t) con ciclo diurno y prediccion de onset.

Cierra la cadena causal de la EPB:

    PRE (E_0 hacia el este al atardecer) -> la capa F sube (E x B)
    + la capa E conjugada recombina      -> F_s -> 1
    => gamma_FT(t) > 0, el crecimiento acumulado int gamma dt llega al nivel
       no lineal => onset (hora local predicha).

Drift vertical empirico tipo Fejer/Scherliess (dia calmo):

    V(t) = V_day cos(2 pi (t-12)/24) + V_pre exp(-(t-t_pre)^2 / 2 w^2)

sinusoide diurna (~20 m/s) + pico PRE gaussiano (~30 m/s extra cerca de
18:45 LT). El campo asociado es E_0x = B V (E hacia el este => deriva E x B
hacia arriba, con B = B y_hat).

La capa E muere al atardecer (recombinacion disociativa sin fotoionizacion);
lo modelo con una logistica en hora local:

    n_E(t) = n_night + (n_day - n_night) / (1 + exp((t - t_sunset)/tau_E))

Trayectoria y crecimiento acumulado:

    dh_peak/dt = V(t)
    gamma(t)   = max_bottomside gamma_FT(h_apex; h_peak(t), n_E(t))
    Gamma(t)   = int max(gamma, 0) dt'

Criterio de onset: Gamma >= ln(10^3) ~ 6.9 e-folds, o sea una semilla del
0.1% amplificada a orden 1 (el criterio usual en los analisis tipo
Sultan/Huba). La hora en que se cruza es la prediccion, comparable con la
ocurrencia observada (19:30-22 LT).

Host-side (NumPy), igual que epb_fluxtube: es el forzante lento (horas) que
alimenta E0x(t), R_E(t) y perfiles a los kernels JAX; no va en el lazo jiteado.
"""

import math
from typing import NamedTuple, Optional

import numpy as np

from PyExner.solvers.kernels.epb_ionosphere import FRegionParams
from PyExner.solvers.kernels.epb_fluxtube import (
    ELayerParams, ELAYER_DAY, ELAYER_NIGHT, gamma_bottomside_max,
)


# --------------------------------------------------------------------------- #
# Parametros del ciclo diurno / PRE                                            #
# --------------------------------------------------------------------------- #

class PREParams(NamedTuple):
    """Ciclo diurno del drift + PRE + decaimiento de la capa E.

    Valores tipicos de epoca de alta ocurrencia (equinoccio, F10.7 moderado):
    drift diurno ~20-30 m/s, pico PRE ~20-60 m/s en ~18:30-19:30 LT (rangos
    de Fejer et al.).
    """
    V_day: float = 20.0          # amplitud de la sinusoide diurna [m/s]
    V_pre: float = 30.0          # amplitud extra del PRE [m/s]
    t_pre: float = 18.75         # hora local del pico PRE [h]
    w_pre: float = 0.5           # anchura gaussiana del PRE [h]
    t_sunset: float = 18.2       # terminador de la capa E [h LT]
    tau_E: float = 0.4           # escala del decaimiento logistico de n_E [h]
    n_E_day: float = ELAYER_DAY.n_max_E      # pico capa E diurna [m^-3]
    n_E_night: float = ELAYER_NIGHT.n_max_E  # remanente nocturno [m^-3]


GROWTH_THRESHOLD = math.log(1.0e3)   # ~6.9 e-folds: semilla 1e-3 -> orden 1


def vertical_drift(t_lt, p: PREParams = PREParams()):
    """Drift vertical de plasma V(t) [m/s] en funcion de la hora local [h]."""
    t = np.asarray(t_lt, dtype=float)
    diurnal = p.V_day * np.cos(2.0 * np.pi * (t - 12.0) / 24.0)
    pre = p.V_pre * np.exp(-((t - p.t_pre) ** 2) / (2.0 * p.w_pre**2))
    return diurnal + pre


def pre_electric_field(t_lt, B: float, p: PREParams = PREParams()):
    """E_0x(t) = B V(t): campo este-oeste que produce la deriva vertical [V/m]."""
    return B * vertical_drift(t_lt, p)


def elayer_density(t_lt, p: PREParams = PREParams()):
    """Pico de la capa E n_E(t): logistica dia -> noche en el terminador."""
    t = np.asarray(t_lt, dtype=float)
    s = 1.0 / (1.0 + np.exp((t - p.t_sunset) / p.tau_E))
    return p.n_E_night + (p.n_E_day - p.n_E_night) * s


def layer_height(t_lt, h0: float, t0: float, p: PREParams = PREParams(),
                 h_floor: float = 200.0e3):
    """Trayectoria del pico F2: h(t) = h0 + int V dt' (trapecio).

    t_lt debe ser una malla creciente que arranca en t0. La capa no baja de
    h_floor: de noche la quimica erosiona el bottomside en vez de seguir
    bajandolo.
    """
    t = np.asarray(t_lt, dtype=float)
    V = vertical_drift(t, p)                       # [m/s]
    dh = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (V[1:] + V[:-1]) * np.diff(t) * 3600.0),
    ])
    return np.maximum(h0 + dh, h_floor)


# --------------------------------------------------------------------------- #
# Prediccion del onset                                                          #
# --------------------------------------------------------------------------- #

class PRENightcast(NamedTuple):
    """Series temporales del pronostico de la tarde-noche (todas de forma (T,))."""
    t_lt: np.ndarray        # hora local [h]
    V: np.ndarray           # drift vertical [m/s]
    h_peak: np.ndarray      # altura del pico F2 [m]
    n_E: np.ndarray         # pico de la capa E [m^-3]
    gamma: np.ndarray       # gamma_FT maximo del bottomside [s^-1]
    Gamma: np.ndarray       # e-folds acumulados int max(gamma,0) dt
    t_onset: Optional[float]   # hora local del onset (None si no se alcanza)


def onset_prediction(p: PREParams = PREParams(),
                     fr: FRegionParams = FRegionParams(),
                     el: ELayerParams = ELAYER_NIGHT,
                     t0: float = 16.0, t1: float = 26.0, dt_lt: float = 2.0 / 60.0,
                     h0: float = 330.0e3, nlat: int = 401) -> PRENightcast:
    """Integra la tarde-noche y predice la hora local del onset.

    En cada instante se reconstruye el perfil F con el pico desplazado
    (subida rigida por E x B, la forma Chapman se conserva: aproximacion
    estandar) y la capa E con su pico decayendo; gamma_bottomside_max da el
    gamma del perfil. El onset es el cruce de Gamma con GROWTH_THRESHOLD.
    """
    t = np.arange(t0, t1 + 1e-9, dt_lt)
    V = vertical_drift(t, p)
    h = layer_height(t, h0, t0, p)
    nE = elayer_density(t, p)

    gam = np.empty_like(t)
    for k in range(len(t)):
        fr_k = fr._replace(h_peak=float(h[k]))
        el_k = el._replace(n_max_E=float(nE[k]))
        gam[k], _ = gamma_bottomside_max(fr_k, el_k, nlat=nlat)

    growth = np.maximum(gam, 0.0)
    Gamma = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (growth[1:] + growth[:-1]) * np.diff(t) * 3600.0),
    ])
    above = np.where(Gamma >= GROWTH_THRESHOLD)[0]
    t_onset = float(t[above[0]]) if len(above) else None
    return PRENightcast(t_lt=t, V=V, h_peak=h, n_E=nE, gamma=gam,
                        Gamma=Gamma, t_onset=t_onset)
