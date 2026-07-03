# PyExner/domain/boundaries/epb_transmissive.py
"""Contorno transmisivo (zero-gradient) para la rama EPB_TwoFluid.

Copia de orden cero: Q_borde = Q_interior, para las 8 componentes. Es la
opcion conservadora para un outflow hiperbolico: deja salir las ondas sin
meter informacion espuria. A diferencia del caso SWE aca no se refleja
ninguna componente de momento; las tres corrientes de ambos fluidos pasan
tal cual.

El nombre registrado sigue la convencion flux_scheme + " " + type
("EPB_TwoFluid Transmissive"), asi BoundaryManager lo instancia igual que
cualquier Transmissive.
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
        """Q_borde = Q_interior para las 8 componentes."""
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
