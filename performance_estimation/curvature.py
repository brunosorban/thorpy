import numpy as np
from matplotlib import pyplot as plt

g = 9.81
m = 100

curvature = lambda f, v: abs(f) / (1 + v**2) ** (3 / 2)

theta = np.linspace(0, 2 * np.pi, 1000)

f = m * g * (2 + np.sin(theta - np.pi))

kappa_15 = curvature(f, 15)
kappa_30 = curvature(f, 30)
kappa_50 = curvature(f, 50)
kappa_100 = curvature(f, 100)
kappa_150 = curvature(f, 150)

plt.plot(theta, 1 / kappa_15, label="v=15")
plt.plot(theta, 1 / kappa_30, label="v=30")
plt.plot(theta, 1 / kappa_50, label="v=50")
plt.plot(theta, 1 / kappa_100, label="v=100")
plt.plot(theta, 1 / kappa_150, label="v=150")
plt.xlabel(r"$\theta$ [rad]")
plt.ylabel(r"$\rho$ [m]")
plt.xticks(np.linspace(0, 2 * np.pi, 5), ["0", "π/2", "π", "3π/2", "2π"])
plt.title("Curvature radius of a backflip as a function of the velocity")
plt.legend()
plt.grid()
plt.show()
