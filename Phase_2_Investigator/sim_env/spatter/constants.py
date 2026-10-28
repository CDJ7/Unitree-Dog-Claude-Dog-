"""Physical constants for droplet flight. SI units throughout (m, s, kg)."""
import numpy as np

G = 9.81
GRAVITY = np.array([0.0, 0.0, -G])
RHO_BLOOD = 1060.0      # kg/m^3
RHO_AIR = 1.2           # kg/m^3
CD_SPHERE = 0.47        # drag coefficient of a sphere at the Reynolds numbers of falling drops

# The five mechanisms the generator and the classifier know about.
MECHANISMS = ("impact", "castoff", "projected", "drip", "transfer")


def drag_k(diameter):
    """k in a_drag = -k |v| v for a sphere of the given diameter (m).

    k = 0.5 rho_air Cd A / m = 3 rho_air Cd / (4 rho_blood d)
    """
    return 3.0 * RHO_AIR * CD_SPHERE / (4.0 * RHO_BLOOD * np.asarray(diameter, dtype=float))
