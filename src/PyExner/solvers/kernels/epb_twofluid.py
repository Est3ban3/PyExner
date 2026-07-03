# PyExner/solvers/kernels/epb_twofluid.py
"""Transporte hiperbolico del modelo de dos fluidos (rama EPB_TwoFluid).

Aca vive solo la parte conservativa del sistema,

    dQ/dt + dF/dx + dG/dz = 0,
    Q = [n_i, n_e, j_ix, j_iy, j_iz, j_ex, j_ey, j_ez]

Las fuentes (colisiones, E, B, g) y el solve de phi van en epb_sources.py;
prefiero no mezclarlas aca para poder validar cada operador por separado.

Cosas a tener presentes al leer este archivo:
  - v_i = +j_i/(e n_i) pero v_e = -j_e/(e n_e). El menos del electron viene
    de definir j_e = -e n_e v_e (densidad de corriente, no de momento); si se
    pierde ese signo todo lo demas sale mal.
  - Cierre isotermo p = n kB T, asi que c_i y c_e son constantes del problema.
  - En el transporte los dos fluidos no se ven entre si: el HLL se aplica por
    bloque (ion / electron), cada uno con sus propias cotas de onda.
"""

from typing import NamedTuple
from functools import partial
import math

import jax
import jax.numpy as jnp

import mpi4jax
from mpi4py import MPI

from PyExner.state.epb_twofluid_state import EPBTwoFluidState, EPB_FIELD_ORDER


# Piso de densidad para evitar division por cero al formar velocidades.
DENS_FLOOR = 1e-12

# Indices de las componentes conservadas por fluido dentro del orden canonico
# EPB_FIELD_ORDER = (n_i, n_e, j_ix, j_iy, j_iz, j_ex, j_ey, j_ez).
ION_IDX = (0, 2, 3, 4)   # n_i, j_ix, j_iy, j_iz
ELE_IDX = (1, 5, 6, 7)   # n_e, j_ex, j_ey, j_ez


class EPBPhysParams(NamedTuple):
    """Constantes fisicas del modelo (cierre isotermo).

    Los defaults son adimensionales (dan c_i = c_e = 1) para validar el
    transporte con numeros comodos. Para correr en SI se cargan desde el
    bloque ``epb`` del YAML (ver from_params); en ese caso usar float64.
    """
    e: float = 1.0       # carga elemental
    kB: float = 1.0      # constante de Boltzmann
    Mi: float = 1.0      # masa del ion
    Me: float = 1.0      # masa del electron
    Ti: float = 1.0      # temperatura ionica (constante, isotermo)
    Te: float = 1.0      # temperatura electronica (constante, isotermo)

    @property
    def ci(self) -> float:
        return math.sqrt(self.kB * self.Ti / self.Mi)

    @property
    def ce(self) -> float:
        return math.sqrt(self.kB * self.Te / self.Me)

    @classmethod
    def from_params(cls, params: dict) -> "EPBPhysParams":
        epb = (params or {}).get("epb", {}) or {}
        defaults = cls()
        return cls(
            e=float(epb.get("e", defaults.e)),
            kB=float(epb.get("kB", defaults.kB)),
            Mi=float(epb.get("Mi", defaults.Mi)),
            Me=float(epb.get("Me", defaults.Me)),
            Ti=float(epb.get("Ti", defaults.Ti)),
            Te=float(epb.get("Te", defaults.Te)),
        )


# --------------------------------------------------------------------------- #
# Helpers de empaquetado                                                       #
# --------------------------------------------------------------------------- #

def stack_state(state: EPBTwoFluidState) -> jax.Array:
    """Empaqueta el estado en un array (..., 8) en el orden canonico."""
    return jnp.stack([getattr(state, name) for name in EPB_FIELD_ORDER], axis=-1)


def unstack_state(arr: jax.Array) -> EPBTwoFluidState:
    """Inverso de ``stack_state``: array (..., 8) -> EPBTwoFluidState."""
    return EPBTwoFluidState(**{name: arr[..., i] for i, name in enumerate(EPB_FIELD_ORDER)})


def _velocities(arr: jax.Array, phys: EPBPhysParams):
    """Velocidades de ambos fluidos a partir de Q (..., 8).

    Devuelve (vix, viy, viz, vex, vey, vez). Ojo con el signo:
    v_i = +j_i/(e n_i) pero v_e = -j_e/(e n_e).
    """
    ni = arr[..., 0]
    ne = arr[..., 1]
    jix, jiy, jiz = arr[..., 2], arr[..., 3], arr[..., 4]
    jex, jey, jez = arr[..., 5], arr[..., 6], arr[..., 7]

    e = phys.e
    eni = e * jnp.maximum(ni, DENS_FLOOR)
    ene = e * jnp.maximum(ne, DENS_FLOOR)

    vix, viy, viz = jix / eni, jiy / eni, jiz / eni
    vex, vey, vez = -jex / ene, -jey / ene, -jez / ene
    return vix, viy, viz, vex, vey, vez


# --------------------------------------------------------------------------- #
# Flujos fisicos F(Q) y G(Q)                                                   #
# --------------------------------------------------------------------------- #

def physical_flux(arr: jax.Array, phys: EPBPhysParams, axis: str) -> jax.Array:
    """Tensor de flujo fisico en la direccion ``axis`` ('x' o 'z').

    Entrada/salida: arrays (..., 8) en el orden canonico de Q.
    """
    ni = arr[..., 0]
    ne = arr[..., 1]
    jix, jiy, jiz = arr[..., 2], arr[..., 3], arr[..., 4]
    jex, jey, jez = arr[..., 5], arr[..., 6], arr[..., 7]

    e = phys.e
    vix, viy, viz, vex, vey, vez = _velocities(arr, phys)

    # Presiones isotermas y su coeficiente (e/M) p.
    api = (e / phys.Mi) * (ni * phys.kB * phys.Ti)   # (e/Mi) p_i
    ape = (e / phys.Me) * (ne * phys.kB * phys.Te)   # (e/Me) p_e

    if axis == "x":
        vin, ven = vix, vex
        # Continuidad: n v_normal = (corriente normal)/e con su signo.
        f_ni = jix / e
        f_ne = -jex / e
    elif axis == "z":
        vin, ven = viz, vez
        f_ni = jiz / e
        f_ne = -jez / e
    else:
        raise ValueError(f"axis must be 'x' or 'z', got {axis!r}")

    # Momento: j_comp * v_normal (+ presion solo en la componente == normal).
    f_jix = jix * vin + (api if axis == "x" else 0.0)
    f_jiy = jiy * vin
    f_jiz = jiz * vin + (api if axis == "z" else 0.0)

    f_jex = jex * ven - (ape if axis == "x" else 0.0)
    f_jey = jey * ven
    f_jez = jez * ven - (ape if axis == "z" else 0.0)

    return jnp.stack(
        [f_ni, f_ne, f_jix, f_jiy, f_jiz, f_jex, f_jey, f_jez], axis=-1
    )


# --------------------------------------------------------------------------- #
# Velocidades de onda (cotas de Davis) y flujo numerico HLL                    #
# --------------------------------------------------------------------------- #

def _wave_bounds(arrL, arrR, phys: EPBPhysParams, axis: str):
    """Cotas de onda tipo Davis, por fluido.

    Devuelve (SL, SR) de forma (..., 8): cada componente lleva las cotas del
    fluido al que pertenece (ion en {0,2,3,4}, electron en {1,5,6,7}).
    """
    vixL, _, vizL, vexL, _, vezL = _velocities(arrL, phys)
    vixR, _, vizR, vexR, _, vezR = _velocities(arrR, phys)

    if axis == "x":
        viL, viR, veL, veR = vixL, vixR, vexL, vexR
    else:  # 'z'
        viL, viR, veL, veR = vizL, vizR, vezL, vezR

    ci, ce = phys.ci, phys.ce

    SiL = jnp.minimum(viL, viR) - ci
    SiR = jnp.maximum(viL, viR) + ci
    SeL = jnp.minimum(veL, veR) - ce
    SeR = jnp.maximum(veL, veR) + ce

    # Colocar las cotas en el orden canonico: ion en {0,2,3,4}, ele en {1,5,6,7}.
    SL = jnp.stack([SiL, SeL, SiL, SiL, SiL, SeL, SeL, SeL], axis=-1)
    SR = jnp.stack([SiR, SeR, SiR, SiR, SiR, SeR, SeR, SeR], axis=-1)
    return SL, SR


def hll_flux(arrL, arrR, phys: EPBPhysParams, axis: str) -> jax.Array:
    """Flujo numerico HLL en la interfaz (vectorizado, arrays (..., 8))."""
    FL = physical_flux(arrL, phys, axis)
    FR = physical_flux(arrR, phys, axis)
    SL, SR = _wave_bounds(arrL, arrR, phys, axis)

    denom = jnp.where(jnp.abs(SR - SL) > 0.0, SR - SL, 1.0)
    F_star = (SR * FL - SL * FR + SL * SR * (arrR - arrL)) / denom

    flux = jnp.where(SL >= 0.0, FL, jnp.where(SR <= 0.0, FR, F_star))
    return flux


# --------------------------------------------------------------------------- #
# Reconstruccion MUSCL de 2do orden (limitador MC)                             #
# --------------------------------------------------------------------------- #
#
# El HLL con reconstruccion constante a trozos es muy difusivo: borra las
# plumas afiladas y subestima el crecimiento RT (lo medi en las pruebas de RT:
# gamma salia ~0.2x del teorico). MUSCL reconstruye un perfil lineal por celda
# con pendiente limitada. Al ser TVD, los valores de cara quedan acotados
# entre vecinos: si las densidades de celda son positivas, las de cara
# tambien. Positividad gratis, sin floor artificial.

def _minmod3(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """minmod de tres: si comparten signo devuelve el de menor magnitud,
    si no cero."""
    s = jnp.sign(a)
    same = (s == jnp.sign(b)) & (s == jnp.sign(c))
    mag = jnp.minimum(jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))
    return jnp.where(same, s * mag, 0.0)


def _mc_slope(Qm: jax.Array, Q0: jax.Array, Qp: jax.Array) -> jax.Array:
    """Pendiente MC (monotonized central) de la celda i.

    slope = minmod( (Q_{i+1}-Q_{i-1})/2,  2(Q_i-Q_{i-1}),  2(Q_{i+1}-Q_i) )

    Menos difusiva que minmod puro y sigue siendo TVD (Sweby con beta = 2).
    Qm, Q0, Qp son Q_{i-1}, Q_i, Q_{i+1}.
    """
    dm = Q0 - Qm        # diferencia hacia atras
    dp = Qp - Q0        # diferencia hacia delante
    central = 0.5 * (Qp - Qm)
    return _minmod3(central, 2.0 * dm, 2.0 * dp)


def _muscl_faces_periodic(Q: jax.Array, axis_np: int):
    """Estados reconstruidos en las caras izquierda/derecha de cada celda.

    Periodico via ``jnp.roll`` en el eje numpy ``axis_np`` (0 = z, 1 = x).
    Devuelve (Q_right, Q_left): el valor en la cara DERECHA (i+1/2^-) y en la
    cara IZQUIERDA (i-1/2^+) de la celda i, ambos de la forma de ``Q``.
    """
    Qm = jnp.roll(Q, 1, axis=axis_np)    # Q_{i-1}
    Qp = jnp.roll(Q, -1, axis=axis_np)   # Q_{i+1}
    slope = _mc_slope(Qm, Q, Qp)
    Q_right = Q + 0.5 * slope            # extrapolacion a la cara i+1/2 (desde i)
    Q_left = Q - 0.5 * slope             # extrapolacion a la cara i-1/2 (desde i)
    return Q_right, Q_left


def muscl_hll_flux_periodic(Q: jax.Array, phys: EPBPhysParams, axis: str) -> jax.Array:
    """Flujo HLL de 2do orden (MUSCL) en TODAS las interfaces, periodico.

    Devuelve el array de flujos alineado de modo que ``flux[i]`` es el flujo en
    la interfaz i+1/2 (entre la celda i y la i+1) a lo largo de ``axis``.
    """
    axis_np = 1 if axis == "x" else 0
    Q_right, Q_left = _muscl_faces_periodic(Q, axis_np)
    # En la interfaz i+1/2: estado izquierdo = cara derecha de la celda i;
    # estado derecho = cara izquierda de la celda i+1.
    QL_iface = Q_right
    QR_iface = jnp.roll(Q_left, -1, axis=axis_np)
    return hll_flux(QL_iface, QR_iface, phys, axis)


# --------------------------------------------------------------------------- #
# Paso de transporte y paso de tiempo                                          #
# --------------------------------------------------------------------------- #

def transport_step_muscl(state: EPBTwoFluidState, dt: float, dx: float, phys: EPBPhysParams) -> EPBTwoFluidState:
    """Paso de transporte de 2do orden (MUSCL + HLL), dominio periodico.

    Reconstruccion lineal limitada en espacio + SSP-RK2 (Heun) en tiempo. La
    RK2 no es opcional: 2do orden espacial con Euler explicito es linealmente
    inestable (me paso al probarlo, en el limite suave el esquema queda
    centrado). Como el paso es combinacion convexa de dos Euler TVD, con
    CFL <= 1 se recupera estabilidad, TVD y positividad de las densidades.

    Misma convencion de signos que ``transport_step`` (actualizacion no-split:
    las diferencias de flujo en x y z se acumulan en un solo update). Dominio
    doblemente periodico, consistente con ``solve_phi``; para contornos
    fisicos falta la capa de ghost cells, por eso el step_fn de produccion
    sigue usando el de 1er orden.
    """
    Q = stack_state(state)
    L0 = -_muscl_divergence(Q, phys, dx)
    Q1 = Q + dt * L0
    L1 = -_muscl_divergence(Q1, phys, dx)
    Q_new = 0.5 * Q + 0.5 * (Q1 + dt * L1)
    return unstack_state(Q_new)


def _muscl_divergence(Q: jax.Array, phys: EPBPhysParams, dx: float) -> jax.Array:
    """Divergencia del flujo MUSCL/HLL (periodica) sobre Q (..., 8): (div_x+div_z)/dx."""
    Fx = muscl_hll_flux_periodic(Q, phys, "x")
    divx = Fx - jnp.roll(Fx, 1, axis=1)
    Gz = muscl_hll_flux_periodic(Q, phys, "z")
    divz = Gz - jnp.roll(Gz, 1, axis=0)
    return (divx + divz) / dx


def transport_step(state: EPBTwoFluidState, dt: float, dx: float, mask, phys: EPBPhysParams) -> EPBTwoFluidState:
    """Un paso dt de transporte por volumenes finitos (HLL, 1er orden).

    Los barridos en x y z se acumulan en la misma divergencia y se aplica un
    solo update (esquema no-split, igual que la rama Roe). Convencion de
    signos: el flujo sale de la celda izquierda (+) y entra a la derecha (-),
    Q^{n+1} = Q^n - (dt/dx) * div.
    """
    Q = stack_state(state)

    div = jnp.zeros_like(Q)

    # --- Barrido en x (axis=1, columnas) ---
    QL_x = Q[:, :-1, :]
    QR_x = Q[:, 1:, :]
    Fx = hll_flux(QL_x, QR_x, phys, axis="x")
    div = div.at[:, :-1, :].add(Fx)   # celda izquierda: salida
    div = div.at[:, 1:, :].add(-Fx)   # celda derecha: entrada

    # --- Barrido en z (axis=0, filas) ---
    QL_z = Q[:-1, :, :]
    QR_z = Q[1:, :, :]
    Gz = hll_flux(QL_z, QR_z, phys, axis="z")
    div = div.at[:-1, :, :].add(Gz)
    div = div.at[1:, :, :].add(-Gz)

    Q_new = Q - (dt / dx) * div
    return unstack_state(Q_new)


@partial(jax.jit, static_argnums=(3,))
def compute_dt_2D(state: EPBTwoFluidState, dx: float, mask: jax.Array, phys: EPBPhysParams) -> float:
    """dt de CFL local (el factor cfl lo aplica el caller).

    dt = min sobre celdas activas de dx / max(|v_n| + c) entre ambos fluidos
    y ambas direcciones. La reduccion global MPI.MIN la hace compute_dt_epb.
    Notar que el electron manda: c_e >> c_i en SI.
    """
    arr = stack_state(state)
    vix, _, viz, vex, _, vez = _velocities(arr, phys)
    ci, ce = phys.ci, phys.ce

    sx = jnp.maximum(jnp.abs(vix) + ci, jnp.abs(vex) + ce)
    sz = jnp.maximum(jnp.abs(viz) + ci, jnp.abs(vez) + ce)
    smax = jnp.maximum(sx, sz)

    # mask = stack([blocked, b_mask]); activas = no bloqueadas.
    active = ~mask[0].astype(bool)
    local = jnp.where(active, dx / (smax + DENS_FLOOR), jnp.inf)
    return jnp.min(local)


def make_halo_exchange(mpi_handler):
    """Intercambio de halos de 1 celda entre subdominios MPI.

    Misma estructura que en las ramas hidraulicas: ronda circular
    west -> north -> east -> south con mpi4jax, jiteable.
    """
    neighbors = mpi_handler.neighbors
    comm = mpi_handler.cart_comm

    @jax.jit
    def halo_exchange(arr):
        send_order = ("west", "north", "east", "south")
        recv_order = ("east", "south", "west", "north")

        overlap_slices_send = dict(
            south=(1, slice(None)),
            west=(slice(None), 1),
            north=(-2, slice(None)),
            east=(slice(None), -2),
        )
        overlap_slices_recv = dict(
            south=(0, slice(None)),
            west=(slice(None), 0),
            north=(-1, slice(None)),
            east=(slice(None), -1),
        )

        for send_dir, recv_dir in zip(send_order, recv_order):
            send_proc = neighbors[send_dir]
            recv_proc = neighbors[recv_dir]

            if send_proc is MPI.PROC_NULL and recv_proc is MPI.PROC_NULL:
                continue

            recv_idx = overlap_slices_recv[recv_dir]
            recv_arr = jnp.empty_like(arr[recv_idx])

            send_idx = overlap_slices_send[send_dir]
            send_arr = arr[send_idx]

            if send_proc is MPI.PROC_NULL:
                recv_arr = mpi4jax.recv(recv_arr, source=recv_proc, comm=comm)
                arr = arr.at[recv_idx].set(recv_arr)
            elif recv_proc is MPI.PROC_NULL:
                mpi4jax.send(send_arr, dest=send_proc, comm=comm)
            else:
                recv_arr = mpi4jax.sendrecv(
                    send_arr,
                    recv_arr,
                    source=recv_proc,
                    dest=send_proc,
                    comm=comm,
                )
                arr = arr.at[recv_idx].set(recv_arr)

        return arr

    return halo_exchange


def halo_exchange_all(state: EPBTwoFluidState, halo_exchange) -> EPBTwoFluidState:
    """Aplica el intercambio de halos a las 8 componentes conservadas de Q."""
    updated = {name: halo_exchange(getattr(state, name)) for name in EPB_FIELD_ORDER}
    return state.replace(**updated)
