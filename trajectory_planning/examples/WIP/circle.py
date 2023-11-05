import sys
import numpy as np
from Traj_planning.traj_generator_3ST import *

sys.path.append("../")
sys.path.append("../Traj_planning")

from parameters import *
from Traj_planning.examples.simple_circ import *

v = 10  # 40 works
r = 100  # with 100

# constraints
max_v = 150
max_a = 30

x, y, vx, vy, t, e1bx, e1by = calculate_traj_params(v, r, m, g)
z = np.zeros_like(x)

constraints = {
    "max_v": max_v,
    "max_a": max_a,
    "g": g,
}

none_vec = np.array([None] * len(x))
none_vec[0] = 0
none_vec[-1] = 0

states = {
    "t": t,
    "x": x,
    "y": y,
    "z": z,
    "vx": vx,
    "vy": vy,
    "vz": none_vec,
    "ax": none_vec,
    "ay": none_vec,
    "az": none_vec,
    "gamma_dot": none_vec,
}


def traj_circle_3ST():
    traj_params = trajenerator_3ST(states, constraints, env_params, rocket_params, controller_params)
    return traj_params
