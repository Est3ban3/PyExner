"""Contorno transmisivo (zero-gradient) para la rama ``EPB_TwoFluid``.

Modelo hiperbolico de dos fluidos (Fase 4). El contorno transmisivo de orden
cero copia las **8 componentes conservadas** desde la primera celda interior a
la celda de borde:

    Q_borde = Q_interior

Es la eleccion conservadora para un flujo de salida (outflow) hiperbolico: no
introduce informacion espuria y deja salir las ondas. A diferencia del caso
hidraulico (SWE), aqui NO se refleja ninguna componente de momento; las tres
corrientes (x, y, z) de ambos fluidos se transmiten tal cual, coherente con un
contorno abierto.

El nombre registrado es ``"EPB_TwoFluid Transmissive"`` (``flux_scheme`` +
``" "`` + ``type``), de modo que ``BoundaryManager`` lo instancia con
``mask``, ``normal``, ``boundary_indices`` e ``interior_indices`` (rama
"Transmissive").
"""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.epb_twofluid_state import EPBTwoFluidState, EPB_FIELD_ORDER


@register_boundary("EPB_TwoFluid Transmissive")
@dataclass
class EPB_TransmissiveBoundary:
    mask: jnp.ndarray                                   # (Ny, Nx), boolean
    normal: jnp.ndarray                                 # (2,), [nx, ny]
    interior_indices: Tuple[jnp.ndarray, jnp.ndarray]   # (y_interior, x_interior)
    boundary_indices: Tuple[jnp.ndarray, jnp.ndarray]   # (y_boundary, x_boundary)

    def apply(self, state: EPBTwoFluidState, time: float) -> EPBTwoFluidState:
        """Copia las 8 componentes de Q de la celda interior a la de borde."""
        by, bx = self.boundary_indices
        iy, ix = self.interior_indices

        new_values = {}
        for name in EPB_FIELD_ORDER:
            field = getattr(state, name)
            new_values[name] = field.at[by, bx].set(field[iy, ix])

        return state.replace(**new_values)


def EPB_TransmissiveBoundary_flatten(b: EPB_TransmissiveBoundary):
    children = (b.mask, b.normal, b.interior_indices, b.boundary_indices)
    return children, None


def EPB_TransmissiveBoundary_unflatten(aux, children):
    return EPB_TransmissiveBoundary(*children)


jax.tree_util.register_pytree_node(
    EPB_TransmissiveBoundary,
    EPB_TransmissiveBoundary_flatten,
    EPB_TransmissiveBoundary_unflatten,
)
