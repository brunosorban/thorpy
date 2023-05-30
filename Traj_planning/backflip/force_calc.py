import numpy as np
import matplotlib.pyplot as plt


def curve(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def curve_derivative(r, theta):
    return -r * np.sin(theta), r * np.cos(theta)


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


def plot_curve(v, r):
    theta = np.linspace(0, 2 * np.pi, 100)

    x, y = curve(r, theta)
    a_cp = calculate_centripetal_acceleration(v, r, theta)

    plt.figure()
    plt.plot(x, y, label="Trajectory")
    plt.quiver(x, y, a_cp[0], a_cp[1], color="red", label="Force Vector", scale=30)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Force Vector over Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_force(v, r, m, g=9.8):
    theta = np.linspace(0, 2 * np.pi, 100)

    x, y = curve(r, theta)
    f = calculate_force(v, r, theta, m, g)

    plt.figure()
    plt.plot(x, y, label="Trajectory")
    plt.quiver(x, y, f[0], f[1], color="red", label="Force Vector", scale=10000)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Force Vector over Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.ylim(-1.1 * r, 1.4 * r)
    plt.show()


# Example usage
v = 300  # m/s
r = 5000  # meters
m = 100  # kg

plot_force(v, r, m)
