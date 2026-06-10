"""Operador de fuentes rigidas y solve electrostatico (rama ``EPB_TwoFluid``).

FASE 5 — Fuentes S(Q) y acoplamiento electrostatico (potencial de polarizacion).

Este modulo es DELIBERADAMENTE SEPARADO del kernel de transporte
(``epb_twofluid.py``): el paso hiperbolico no debe mezclar la dinamica de
transporte con colisiones ni con el solve de phi (principio de separacion del
plan de extension).

Contenido:
    * ``EPBSourceParams``  : parametros de fondo (g, E_0, B, U, nu_in, nu_en, nu_ei).
    * ``source_term``      : vector fuente S(Q) (filas de continuidad = 0).
    * ``solve_phi``        : solve eliptico de Poisson para el potencial de
                             polarizacion (estilo proyeccion, ∇²φ = ∇·J).
    * ``electric_field``   : E = E_0 - ∇φ (solo corrige E_x, E_z; E_y fuera del plano).

Modelo continuo de las fuentes (las filas de continuidad NO tienen fuente):

    Las variables transportadas j_alpha son densidades de CORRIENTE
    (j_i = e n_i v_i,  j_e = -e n_e v_e). La ecuacion de momento de cada especie
    multiplicada por q_alpha/M_alpha (mismo escalado que el flujo ±(e/M)p) da:

        S_{j_i} =  e n_i g + (e^2 n_i / M_i) E + (e/M_i) j_i × B
                   - nu_in (j_i - e n_i U)
        S_{j_e} = -e n_e g + (e^2 n_e / M_e) E - (e/M_e) j_e × B
                   - nu_en (j_e + e n_e U) - nu_ei (j_e + (n_e/n_i) j_i)

    con q_i = +e, q_e = -e.

Acoplamiento electrostatico (estilo proyeccion):

    Se impone continuidad de corriente ∇·J = 0 con J = j_i + j_e (componentes en
    el plano x,z). Resolviendo ∇²φ = ∇·J se obtiene el potencial de
    polarizacion, y el campo efectivo es E = E_0 - ∇φ.

Orden de actualizacion previsto (la integracion rigida es la Fase 6 / IMEX):

    transporte  ->  solve φ (desde el estado)  ->  E = E_0 - ∇φ  ->  fuentes
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from functools import partial

from PyExner.solvers.kernels.epb_twofluid import (
    EPBPhysParams,
    stack_state,
    unstack_state,
    DENS_FLOOR,
)


class EPBSourceParams(NamedTuple):
    """Parametros de fondo del operador de fuentes (electrodinamica + colisiones).

    Geometria 2.5D (plano x,z): los vectores de fondo tienen tres componentes,
    pero solo x y z se discretizan espacialmente. Valores por defecto NULOS
    (sin fuentes) para que el transporte puro de la Fase 3 sea el caso por
    defecto y no se altere ningun resultado previo.
    """
    # Gravedad g = (gx, gy, gz)
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    # Campo electrico de fondo E_0 = (E0x, E0y, E0z)
    E0x: float = 0.0
    E0y: float = 0.0
    E0z: float = 0.0
    # Campo magnetico B = (Bx, By, Bz). En geometria ecuatorial suele ir fuera
    # del plano (eje y); por defecto nulo.
    Bx: float = 0.0
    By: float = 0.0
    Bz: float = 0.0
    # Viento neutro de fondo U = (Ux, Uy, Uz)
    Ux: float = 0.0
    Uy: float = 0.0
    Uz: float = 0.0
    # Frecuencias de colision
    nu_in: float = 0.0   # ion - neutro
    nu_en: float = 0.0   # electron - neutro
    nu_ei: float = 0.0   # electron - ion
    # Iteraciones del solve de Poisson (parametro numerico, no fisico).
    poisson_iters: int = 200

    @property
    def g(self):
        return jnp.array([self.gx, self.gy, self.gz])

    @property
    def E0(self):
        return jnp.array([self.E0x, self.E0y, self.E0z])

    @property
    def B(self):
        return jnp.array([self.Bx, self.By, self.Bz])

    @property
    def U(self):
        return jnp.array([self.Ux, self.Uy, self.Uz])

    @classmethod
    def from_params(cls, params: dict) -> "EPBSourceParams":
        src = ((params or {}).get("epb", {}) or {}).get("sources", {}) or {}
        d = cls()
        return cls(
            gx=float(src.get("gx", d.gx)), gy=float(src.get("gy", d.gy)), gz=float(src.get("gz", d.gz)),
            E0x=float(src.get("E0x", d.E0x)), E0y=float(src.get("E0y", d.E0y)), E0z=float(src.get("E0z", d.E0z)),
            Bx=float(src.get("Bx", d.Bx)), By=float(src.get("By", d.By)), Bz=float(src.get("Bz", d.Bz)),
            Ux=float(src.get("Ux", d.Ux)), Uy=float(src.get("Uy", d.Uy)), Uz=float(src.get("Uz", d.Uz)),
            nu_in=float(src.get("nu_in", d.nu_in)),
            nu_en=float(src.get("nu_en", d.nu_en)),
            nu_ei=float(src.get("nu_ei", d.nu_ei)),
            poisson_iters=int(src.get("poisson_iters", d.poisson_iters)),
        )


# --------------------------------------------------------------------------- #
# Vector fuente S(Q)                                                           #
# --------------------------------------------------------------------------- #

def source_term(arr: jax.Array, phys: EPBPhysParams, src: EPBSourceParams, E: jax.Array) -> jax.Array:
    """Vector fuente S(Q) de forma (..., 8) en el orden canonico.

    Las filas de continuidad (n_i, n_e) son CERO: las fuentes solo actuan sobre
    las densidades de corriente. ``E`` es el campo electrico efectivo por celda,
    de forma (..., 3) (tipicamente E = E_0 - ∇φ).
    """
    ni = arr[..., 0]
    ne = arr[..., 1]
    ji = arr[..., 2:5]   # (jix, jiy, jiz)
    je = arr[..., 5:8]   # (jex, jey, jez)

    e, Mi, Me = phys.e, phys.Mi, phys.Me
    g, B, U = src.g, src.B, src.U

    ni_ = ni[..., None]
    ne_ = ne[..., None]

    # --- Bloque ionico ---
    S_i = (
        (e * ni_) * g                       # gravedad
        + (e * e * ni_ / Mi) * E            # fuerza electrica
        + (e / Mi) * jnp.cross(ji, B)       # fuerza magnetica
        - src.nu_in * (ji - e * ni_ * U)    # colision ion-neutro
    )

    # --- Bloque electronico ---
    ratio = ne_ / jnp.maximum(ni_, DENS_FLOOR)
    S_e = (
        (-e * ne_) * g                      # gravedad
        + (e * e * ne_ / Me) * E            # fuerza electrica
        - (e / Me) * jnp.cross(je, B)       # fuerza magnetica
        - src.nu_en * (je + e * ne_ * U)    # colision electron-neutro
        - src.nu_ei * (je + ratio * ji)     # colision electron-ion
    )

    zeros = jnp.zeros_like(ni)
    return jnp.stack(
        [
            zeros, zeros,
            S_i[..., 0], S_i[..., 1], S_i[..., 2],
            S_e[..., 0], S_e[..., 1], S_e[..., 2],
        ],
        axis=-1,
    )


# --------------------------------------------------------------------------- #
# Solve electrostatico: potencial de polarizacion y campo efectivo            #
# --------------------------------------------------------------------------- #

def _central_diff(field: jax.Array, dx: float, axis: int) -> jax.Array:
    """Derivada central de 2do orden con bordes periodicos.

    OBSOLETA para el solve electrostatico (introduce desacoplamiento par/impar:
    ``div_c o grad_c`` != laplaciano de 5 puntos). Se conserva por compatibilidad
    de diagnostico; el solve de produccion usa la tripleta consistente
    ``_div_backward`` / ``_grad_forward`` / ``_laplacian`` (Paso 4).
    """
    fwd = jnp.roll(field, -1, axis=axis)
    bwd = jnp.roll(field, 1, axis=axis)
    return (fwd - bwd) / (2.0 * dx)


# --------------------------------------------------------------------------- #
# Tripleta de operadores CONSISTENTE (Paso 4): D^- G^+ = L_5pt exacto.         #
#                                                                             #
#   grad forward   G^+ f|_i  = (f_{i+1} - f_i)/dx        (cara i+1/2)          #
#   div  backward  D^- u|_i  = (u_i - u_{i-1})/dx        (adjunta de -G^+)     #
#   laplaciano     L  = D^- G^+  =>  (f_{i+1}-2f_i+f_{i-1})/dx^2  (5 puntos)   #
#                                                                             #
# Con esta tripleta la limpieza de divergencia es EXACTA a precision de        #
# maquina: D^-(J - G^+ phi) = D^-J - L phi = residual del CG ~ 0. Esto elimina #
# el modo checkerboard que aparecia al mezclar laplaciano de 5 puntos con      #
# gradiente central (hallazgo del Nivel 3).                                    #
# --------------------------------------------------------------------------- #

def _grad_forward(field: jax.Array, dx: float, axis: int) -> jax.Array:
    """Gradiente forward G^+ f|_i = (f_{i+1} - f_i)/dx (periodico)."""
    return (jnp.roll(field, -1, axis=axis) - field) / dx


def _div_backward(field: jax.Array, dx: float, axis: int) -> jax.Array:
    """Divergencia backward D^- u|_i = (u_i - u_{i-1})/dx (adjunta de -G^+)."""
    return (field - jnp.roll(field, 1, axis=axis)) / dx


def _laplacian(field: jax.Array, dx: float) -> jax.Array:
    """Laplaciano compacto de 5 puntos = D^- G^+ (axis 0 = z, axis 1 = x).

    Identidad EXACTA con la tripleta consistente:
        _div_backward(_grad_forward(f, x), x) + _div_backward(_grad_forward(f, z), z).
    """
    up = jnp.roll(field, -1, axis=0)
    down = jnp.roll(field, 1, axis=0)
    right = jnp.roll(field, -1, axis=1)
    left = jnp.roll(field, 1, axis=1)
    return (up + down + left + right - 4.0 * field) / (dx * dx)


def _poisson_divergence(arr: jax.Array, dx: float) -> jax.Array:
    """RHS del Poisson: ∇·J con divergencia BACKWARD consistente.

    J = j_i + j_e en el plano (x = indice 2/5, z = indice 4/7). Se proyecta al
    subespacio de media cero (compatibilidad del sistema singular periodico).
    """
    Jx = arr[..., 2] + arr[..., 5]   # jix + jex
    Jz = arr[..., 4] + arr[..., 7]   # jiz + jez
    rhs = _div_backward(Jx, dx, axis=1) + _div_backward(Jz, dx, axis=0)
    return rhs - jnp.mean(rhs)


@partial(jax.jit, static_argnums=(2,))
def _poisson_cg(rhs: jax.Array, dx: float, n_iter: int) -> jax.Array:
    """Gradiente conjugado para -L φ = -rhs, con L el laplaciano de 5 puntos.

    El operador A = -L es simetrico semidefinido positivo (nucleo = constantes
    bajo BC periodicas). Se trabaja en el subespacio de media cero proyectando
    A p -> A p - <A p> en cada producto matriz-vector, y se fija el gauge
    <φ> = 0 al final. CG converge en pocas iteraciones (mucho mejor que Jacobi)
    y es jit-able con ``fori_loop`` de longitud estatica.

    Robustez: una vez convergido (residual relativo < 1e-12) la iteracion se
    CONGELA (alpha=beta=0). Sin esto, iterar de mas tras la convergencia divide
    por ~0 y desestabiliza la solucion; con la congelacion, sobre-especificar
    ``n_iter`` es inocuo.
    """
    def Aop(p):
        ap = -_laplacian(p, dx)
        return ap - jnp.mean(ap)

    b = -(rhs - jnp.mean(rhs))
    phi = jnp.zeros_like(rhs)
    r = b - Aop(phi)          # = b (phi0 = 0)
    p = r
    rs = jnp.sum(r * r)
    rs0 = jnp.maximum(rs, 1e-300)

    def body(_, carry):
        phi, r, p, rs = carry
        Ap = Aop(p)
        denom = jnp.sum(p * Ap)
        # Activo solo mientras NO se ha convergido (residual relativo) y el
        # denominador es positivo (curvatura util).
        active = (rs > 1e-24 * rs0) & (denom > 1e-300)
        alpha = jnp.where(active, rs / jnp.where(denom > 0.0, denom, 1.0), 0.0)
        phi = phi + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.sum(r * r)
        beta = jnp.where(active, rs_new / jnp.where(rs > 0.0, rs, 1.0), 0.0)
        p = jnp.where(active, r + beta * p, p)
        rs_out = jnp.where(active, rs_new, rs)
        return (phi, r, p, rs_out)

    phi, r, p, rs = jax.lax.fori_loop(0, n_iter, body, (phi, r, p, rs))
    return phi - jnp.mean(phi)


@partial(jax.jit, static_argnums=(2,))
def solve_phi(arr: jax.Array, dx: float, n_iter: int) -> jax.Array:
    """Potencial de polarizacion φ resolviendo ∇²φ = ∇·J (estilo proyeccion).

    Paso 4 (produccion): tripleta de operadores CONSISTENTE
    (``_div_backward`` / ``_grad_forward`` / laplaciano de 5 puntos) resuelta por
    gradiente conjugado. ``n_iter`` es el numero (estatico) de iteraciones de CG.
    Convencion de ejes: axis 1 = x, axis 0 = z. BC periodicas (``jnp.roll``).

    Garantia clave: el campo corregido J - ∇φ (con ∇ = ``_grad_forward``) es
    DIVERGENTE-LIBRE a precision de maquina en el sentido de ``_div_backward``,
    porque D^-(J - G^+φ) = rhs - Lφ = residual del CG. No hay modo checkerboard.

    NOTA: para contornos fisicos no periodicos (Neumann/Dirichlet) o un dominio
    distribuido en MPI, el CG debe intercalar ``halo_exchange`` en cada producto
    matriz-vector y ajustar el operador en los bordes; la tripleta consistente y
    el solver CG no cambian.
    """
    rhs = _poisson_divergence(arr, dx)
    return _poisson_cg(rhs, dx, n_iter)


def electric_field(phi: jax.Array, src: EPBSourceParams, dx: float) -> jax.Array:
    """Campo electrico efectivo E = E_0 - ∇φ, de forma (..., 3).

    Usa el gradiente FORWARD (``_grad_forward``), consistente con la divergencia
    backward del solve: asi la correccion limpia exactamente la divergencia
    (sin checkerboard). Solo se corrigen las componentes en el plano (E_x, E_z);
    E_y (fuera del plano) queda en su valor de fondo E_{0y}.
    """
    dphidx = _grad_forward(phi, dx, axis=1)
    dphidz = _grad_forward(phi, dx, axis=0)

    Ex = src.E0x - dphidx
    Ey = jnp.full_like(phi, src.E0y)
    Ez = src.E0z - dphidz
    return jnp.stack([Ex, Ey, Ez], axis=-1)


# --------------------------------------------------------------------------- #
# Paso 5: solve electrostatico con CONDUCTANCIA VARIABLE (carga de capa E)     #
#                                                                             #
# Fisica: la ecuacion de potencial integrada en el tubo de flujo es           #
#                                                                             #
#     div( (Sigma_F + Sigma_E) grad phi ) = div J_drive                       #
#                                                                             #
# con Sigma_F ~ n (conductancia Pedersen local de la region F) y Sigma_E la   #
# conductancia de las capas E conjugadas, que actua como CORTOCIRCUITO en     #
# paralelo: drena la carga de polarizacion a lo largo de B y DILUYE el campo  #
# E_p por el factor de apantallamiento F_s = Sigma_F/(Sigma_F + Sigma_E).     #
# Es el mecanismo por el que la EPB solo crece tras la puesta de sol (la capa #
# E diurna apantalla; la nocturna no).                                        #
#                                                                             #
# Numerica: se generaliza la tripleta consistente del Paso 4 a coeficiente    #
# variable con sigma evaluada en CARAS (media aritmetica):                    #
#                                                                             #
#     L_sigma phi = D^-( sigma_{i+1/2} G^+ phi )                              #
#                                                                             #
# A = -L_sigma sigue siendo simetrica (sigma_f compartida por la pareja de    #
# celdas) y semidefinida positiva para sigma > 0, asi que el MISMO CG del     #
# Paso 4 aplica sin cambios. Con sigma uniforme = c se recupera EXACTAMENTE   #
# phi_{Paso4}/c (shunt analitico). Normalizacion: sigma es adimensional,      #
# sigma = Sigma_total / Sigma_ref con Sigma_ref la conductancia F de          #
# referencia; el caso sigma = 1 reproduce el solve del Paso 4.                #
# --------------------------------------------------------------------------- #

SIGMA_FLOOR = 1e-12   # cota inferior de sigma: preserva SPD si n -> 0 (burbuja)


def _face_avg(sigma: jax.Array, axis: int) -> jax.Array:
    """sigma en la cara i+1/2: media aritmetica (sigma_i + sigma_{i+1})/2."""
    return 0.5 * (sigma + jnp.roll(sigma, -1, axis=axis))


def _div_sigma_grad(phi: jax.Array, sigma: jax.Array, dx: float) -> jax.Array:
    """Operador de coeficiente variable L_sigma phi = D^-(sigma_f G^+ phi).

    Reduce EXACTAMENTE a ``_laplacian`` cuando sigma = 1 (tripleta del Paso 4).
    Simetrico (sigma de cara compartida) y definido negativo en el subespacio
    de media cero para sigma > 0.
    """
    fx = _face_avg(sigma, 1) * _grad_forward(phi, dx, axis=1)
    fz = _face_avg(sigma, 0) * _grad_forward(phi, dx, axis=0)
    return _div_backward(fx, dx, axis=1) + _div_backward(fz, dx, axis=0)


@partial(jax.jit, static_argnums=(3,))
def _poisson_cg_sigma(rhs: jax.Array, sigma: jax.Array, dx: float, n_iter: int) -> jax.Array:
    """CG para -L_sigma phi = -rhs (mismo esquema y guardas que ``_poisson_cg``).

    Identica estructura: subespacio de media cero, gauge <phi> = 0 y
    CONGELACION al converger (residual relativo < 1e-12). Se mantiene como
    funcion separada para no tocar el kernel validado del Paso 4.
    """
    sig = jnp.maximum(sigma, SIGMA_FLOOR)

    def Aop(p):
        ap = -_div_sigma_grad(p, sig, dx)
        return ap - jnp.mean(ap)

    b = -(rhs - jnp.mean(rhs))
    phi = jnp.zeros_like(rhs)
    r = b - Aop(phi)
    p = r
    rs = jnp.sum(r * r)
    rs0 = jnp.maximum(rs, 1e-300)

    def body(_, carry):
        phi, r, p, rs = carry
        Ap = Aop(p)
        denom = jnp.sum(p * Ap)
        active = (rs > 1e-24 * rs0) & (denom > 1e-300)
        alpha = jnp.where(active, rs / jnp.where(denom > 0.0, denom, 1.0), 0.0)
        phi = phi + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.sum(r * r)
        beta = jnp.where(active, rs_new / jnp.where(rs > 0.0, rs, 1.0), 0.0)
        p = jnp.where(active, r + beta * p, p)
        rs_out = jnp.where(active, rs_new, rs)
        return (phi, r, p, rs_out)

    phi, r, p, rs = jax.lax.fori_loop(0, n_iter, body, (phi, r, p, rs))
    return phi - jnp.mean(phi)


@partial(jax.jit, static_argnums=(3,))
def solve_phi_sigma(arr: jax.Array, sigma: jax.Array, dx: float, n_iter: int) -> jax.Array:
    """Potencial de polarizacion con conductancia variable: D^-(sigma G^+ phi) = D^- J.

    Generalizacion del ``solve_phi`` del Paso 4 al cierre con carga de capa E:
    ``sigma`` (adimensional, normalizada a la conductancia F de referencia) es
    tipicamente  sigma = n/n_ref + R_E  con R_E = Sigma_E/Sigma_ref el shunt.
    Garantia heredada: D^-J - L_sigma phi = residual CG ~ 0 (limpieza exacta de
    la divergencia con la corriente corregida sigma_f G^+ phi). BC periodicas.
    """
    rhs = _poisson_divergence(arr, dx)
    return _poisson_cg_sigma(rhs, sigma, dx, n_iter)


# --------------------------------------------------------------------------- #
# Aplicacion explicita (solo para pruebas / diagnostico de la Fase 5)         #
# --------------------------------------------------------------------------- #

def apply_sources_explicit(
    state, dt: float, phys: EPBPhysParams, src: EPBSourceParams, dx: float
):
    """Paso de fuentes Euler explicito Q <- Q + dt S(Q).

    SOLO para validar el operador de fuentes y el acoplamiento de φ de forma
    aislada. NO se usa en ``step_fn`` porque las fuentes pueden ser rigidas
    (colisiones rapidas) y su integracion estable es responsabilidad del
    integrador IMEX (Fase 6). El orden interno respeta la separacion:
        solve φ  ->  E = E_0 - ∇φ  ->  S(Q)  ->  update.
    """
    arr = stack_state(state)
    phi = solve_phi(arr, dx, src.poisson_iters)
    E = electric_field(phi, src, dx)
    S = source_term(arr, phys, src, E)
    return unstack_state(arr + dt * S)


# --------------------------------------------------------------------------- #
# Solve IMPLICITO de fuentes rigidas (Fase 6 / IMEX)                           #
# --------------------------------------------------------------------------- #

def _cross_matrix(Bx: float, By: float, Bz: float) -> jax.Array:
    """Matriz K tal que K j = j × B (operador lineal del producto cruz)."""
    return jnp.array(
        [
            [0.0, Bz, -By],
            [-Bz, 0.0, Bx],
            [By, -Bx, 0.0],
        ]
    )


def implicit_source_solve(
    state, dt: float, phys: EPBPhysParams, src: EPBSourceParams, dx: float,
    sigma: jax.Array | None = None, E_ext: jax.Array | None = None,
):
    """Actualizacion IMPLICITA de las fuentes rigidas (parte ``Im`` del IMEX).

    El vector fuente es AFIN en las corrientes (las densidades no se sourcean):

        S(j) = A j + b

    con (por bloque, q_i=+e, q_e=-e):

        A_ion = (e/M_i) K - nu_in I
        A_ee  = -(e/M_e) K - (nu_en + nu_ei) I
        A_ei  = -nu_ei (n_e/n_i) I            (acoplamiento electron <- ion)

        b_i =  e n_i g + (e^2 n_i/M_i) E + nu_in e n_i U
        b_e = -e n_e g + (e^2 n_e/M_e) E - nu_en e n_e U

    donde K j = j × B. El campo E se evalua de forma LAGGED (congelada) a partir
    del solve eliptico del estado actual: la rigidez vive en las colisiones y en
    la girofrecuencia magnetica (parte implicita local por celda), mientras que
    el acoplamiento electrostatico global se trata explicito. Asi, por celda se
    resuelve el sistema lineal 6x6

        (I - dt A) j^{n+1} = j^n + dt b

    de forma vectorizada con ``jnp.linalg.solve``. La estructura es bloque-
    triangular inferior (el ion no depende del electron), por lo que el sistema
    esta bien condicionado para dt y frecuencias colisionales razonables.

    Paso 5: si se pasa ``sigma`` (conductancia adimensional, p.ej.
    ``n_i/n_ref + R_E`` con el shunt de capa E), el potencial se resuelve con
    ``solve_phi_sigma``; con ``sigma=None`` se conserva el solve del Paso 4
    (sin cambio de comportamiento para los llamadores existentes).

    ``E_ext``: campo electrico TOTAL precalculado, de forma (..., 3). Si se
    pasa, NO se resuelve phi aqui: el llamador es responsable del cierre
    electrostatico (p.ej. el cierre drive-driven de ``plume_si_epb``, donde el
    RHS del solve es solo la corriente motriz y no la corriente total — el
    solve interno con J total es inestable cuando el estado ya contiene la
    respuesta Pedersen sigma_P E^n del paso anterior).
    """
    arr = stack_state(state)
    ni = arr[..., 0]
    ne = arr[..., 1]
    ji = arr[..., 2:5]
    je = arr[..., 5:8]

    # Campo electrico lagged (acoplamiento electrostatico explicito).
    if E_ext is not None:
        E = E_ext
    elif sigma is None:
        phi = solve_phi(arr, dx, src.poisson_iters)
        E = electric_field(phi, src, dx)   # (..., 3)
    else:
        phi = solve_phi_sigma(arr, sigma, dx, src.poisson_iters)
        E = electric_field(phi, src, dx)   # (..., 3)

    e, Mi, Me = phys.e, phys.Mi, phys.Me
    K = _cross_matrix(src.Bx, src.By, src.Bz)   # (3, 3)
    I3 = jnp.eye(3)

    A_ion = (e / Mi) * K - src.nu_in * I3                      # (3, 3) constante
    A_ee = -(e / Me) * K - (src.nu_en + src.nu_ei) * I3        # (3, 3) constante
    ratio = ne / jnp.maximum(ni, DENS_FLOOR)                   # (...)
    A_ei = -src.nu_ei * ratio[..., None, None] * I3            # (..., 3, 3)

    shp = ni.shape
    A = jnp.zeros(shp + (6, 6))
    A = A.at[..., 0:3, 0:3].set(A_ion)
    A = A.at[..., 3:6, 3:6].set(A_ee)
    A = A.at[..., 3:6, 0:3].set(A_ei)

    g, U = src.g, src.U
    ni_ = ni[..., None]
    ne_ = ne[..., None]
    b_i = e * ni_ * g + (e * e * ni_ / Mi) * E + src.nu_in * e * ni_ * U
    b_e = -e * ne_ * g + (e * e * ne_ / Me) * E - src.nu_en * e * ne_ * U
    b = jnp.concatenate([b_i, b_e], axis=-1)        # (..., 6)

    j_old = jnp.concatenate([ji, je], axis=-1)      # (..., 6)
    I6 = jnp.eye(6)
    M = I6 - dt * A
    rhs = (j_old + dt * b)[..., None]               # (..., 6, 1)
    j_new = jnp.linalg.solve(M, rhs)[..., 0]        # (..., 6)

    new = arr.at[..., 2:8].set(j_new)
    return unstack_state(new)

