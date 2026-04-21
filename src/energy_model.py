from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class VehicleParams:
    m: float = 1500.0
    eta: float = 0.3
    fr: float = 0.015
    g: float = 9.81
    b: float = 1.0
    rt: float = 0.3
    r: float = 0.11
    k: float = 0.85
    pa: float = 2000.0
    ro: float = 1.2
    c: float = 0.3
    af: float = 1.8

    @property
    def alpha(self) -> float:
        return 0.5 * self.ro * self.c * self.af


def get_poly_coeffs(speed: float, acceleration: float, params: VehicleParams | None = None) -> List[float]:
    """Return polynomial coefficients [v4, v3, v2, v1, v0] for EV power.

    The implementation preserves the same structure used in the original code.
    The `speed` argument is accepted for API compatibility even though the
    coefficient computation only depends on acceleration and fixed parameters.
    """
    _ = speed
    p = params or VehicleParams()

    v_4 = (p.r * (p.rt / p.k) ** 2) * p.alpha**2
    v_3 = p.alpha + p.r * (p.rt / p.k) ** 2 * (2 * p.alpha * p.b / p.rt)
    v_2 = p.b / p.rt + p.r * (p.rt / p.k) ** 2 * (
        (p.b / p.rt) ** 2 + 2 * p.m * p.alpha * acceleration + 2 * p.alpha * p.fr * p.m * p.g
    )
    v_1 = p.fr * p.m * p.g + p.r * (p.rt / p.k) ** 2 * (
        2 * p.m * acceleration * p.b / p.rt + 2 * p.m * p.fr * p.g * p.b / p.rt
    )
    v_0 = p.r * (p.rt / p.k) ** 2 * (
        p.m**2 * acceleration**2
        + (p.fr * p.m * p.g) ** 2
        + 2 * p.m**2 * p.fr * p.g * acceleration
        + 2 * p.m * p.fr * p.g
    )
    return [v_4, v_3, v_2, v_1, v_0]
