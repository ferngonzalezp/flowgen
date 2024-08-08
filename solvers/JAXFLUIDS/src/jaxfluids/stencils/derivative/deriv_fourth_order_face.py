#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class DerivativeFourthOrderFace(SpatialDerivative):
    ''' 
    4th order stencil for 1st derivative at the cell face
              x
    |     |   |     |     |
    | i-1 | i | i+1 | i+2 |
    |     |   |     |     |
    '''
    def __init__(self, nh: int, inactive_axis: List, offset: int = 0) -> None:
        super(DerivativeFourthOrderFace, self).__init__(nh=nh, inactive_axis=inactive_axis, offset=offset)

        self.s_ = [

            [   jnp.s_[..., self.n-2:-self.n-1, self.nhy, self.nhz],         # i-1
                jnp.s_[..., self.n-1:-self.n  , self.nhy, self.nhz],         # i
                jnp.s_[..., self.n  :-self.n+1, self.nhy, self.nhz],         # i+1
                jnp.s_[..., jnp.s_[self.n+1:-self.n+2] if self.n != 2 else jnp.s_[self.n+1:None], self.nhy, self.nhz]  ],      # i+2

            [   jnp.s_[..., self.nhx, self.n-2:-self.n-1, self.nhz],     
                jnp.s_[..., self.nhx, self.n-1:-self.n  , self.nhz],     
                jnp.s_[..., self.nhx, self.n  :-self.n+1, self.nhz],     
                jnp.s_[..., self.nhx, jnp.s_[self.n+1:-self.n+2] if self.n != 2 else jnp.s_[self.n+1:None], self.nhz]  ],   

            [   jnp.s_[..., self.nhx, self.nhy, self.n-2:-self.n-1],     
                jnp.s_[..., self.nhx, self.nhy, self.n-1:-self.n  ],     
                jnp.s_[..., self.nhx, self.nhy, self.n  :-self.n+1],     
                jnp.s_[..., self.nhx, self.nhy, jnp.s_[self.n+1:-self.n+2] if self.n != 2 else jnp.s_[self.n+1:None]]  ]   
        ]

    def derivative_xi(self, primes: jnp.ndarray, dxi: jnp.ndarray, axis: int) -> jnp.ndarray:
        s1_ = self.s_[axis]
        deriv_xi = (1.0 / 24.0 / dxi) * (primes[s1_[0]] - 27.0 * primes[s1_[1]] + 27.0 * primes[s1_[2]] - primes[s1_[3]])
        return deriv_xi