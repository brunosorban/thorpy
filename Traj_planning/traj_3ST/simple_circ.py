import numpy as np
import matplotlib.pyplot as plt
# from Traj_planning.traj_generators.traj_planner_SE3 import *


def curve(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def vel_direction(theta):
    return np.array([-np.sin(theta), np.cos(theta)])


def calculate_centripetal_acceleration(v, r, theta):
    # Calculate centripetal acceleration
    x, y = curve(r, theta)
    radial = -np.array([x, y]) / np.linalg.norm(np.array([x, y]))
    a_cp = (v**2) / r * radial

    return a_cp


def calculate_force(v, r, theta, m, g=9.8):
    # Calculate centripetal acceleration in the world frame
    a_cp = calculate_centripetal_acceleration(v, r, theta)
    mg = np.zeros_like(a_cp)
    mg[1] = m * g

    # Calculate the force in the world frame
    f = m * a_cp + mg

    return f


def calculate_traj_params(v, r, m, g=9.8):
    """
    Given the velocity, radius, and mass of the rocket, calculate the
    trajectory parameters. The rocket shall go h=2r meters above the ground, reaching
    the point (0, h) with velocity v. It shall do a circular motion of radius r
    from this point starting at 90 degrees and ending at 270 degrees.
    """

    x = []
    y = []
    vx = []
    vy = []
    t = []
    e1bx = []
    e1by = []

    theta = np.linspace(0, 2.5 * np.pi, 10, endpoint=False)

    d_theta = theta[1] - theta[0]
    d_arc = r * d_theta
    dt = d_arc / v
    h = 2 * r - d_arc

    x.append(2 * r)
    y.append(0)
    vx.append(0)
    vy.append(0.8 * v)
    t.append(0)
    e1bx.append(0)
    e1by.append(1)

    x.append(2 * r)
    y.append(h)
    vx.append(0)
    vy.append(v)
    t.append(h / v)
    e1bx.append(0)
    e1by.append(1)

    for i in range(len(theta)):
        x.append(curve(r, theta[i])[0] + r)
        y.append(curve(r, theta[i])[1] + 2 * r)

        vx.append(v * vel_direction(theta[i])[0])
        vy.append(v * vel_direction(theta[i])[1])

        t.append(t[-1] + dt)

        f = calculate_force(v, r, theta[i], m, g)
        e1bx.append(f[0] / np.linalg.norm(f))
        e1by.append(f[1] / np.linalg.norm(f))

    x = np.array(x)
    y = np.array(y)
    vx = np.array(vx)
    vy = np.array(vy)
    t = np.array(t)

    return x, y, vx, vy, t, e1bx, e1by