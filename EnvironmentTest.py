import numpy as np
import matplotlib.pyplot as plt
import rocketDQN

# Test of the ODE and step() + reset() fcn
duration = 20
N = 200

env = rocketDQN.HopperEnv()

state = env.reset()
print(f"The initial state of the hopper is: {state}")

S = []
t = np.linspace(0, duration, N)

for i in enumerate(t):
    new_state, reward, done = env.step(59, state)
    state = new_state
    S.append(new_state[0])

print(S[199])
plt.plot(t, S)
plt.show()

# Comparison plot of Runge-Kutta solver with various step sizes in comparison to the analytical solution
step_size = 0.1
duration = 20
N = duration/step_size
env = rocketDQN.HopperEnv()

state = env.reset()
x0 = state[0]
print(f"The initial state of the hopper is: {state}")

S = []
S_kutta = []
t_kutta = np.linspace(0, duration, int(N))

for i in enumerate(t_kutta):
    new_state, reward, done = env.step(0, state)
    state = new_state
    x = -0.5 * 9.81 * t_kutta[i[0]]**2 + x0
    S.append(x)
    S_kutta.append(new_state[0])

plt.plot(t_kutta, S_kutta)
plt.plot(t_kutta, S)
print(f"Kutta last value: {S_kutta[-1]}")
print(f"Analytical last value: {S[-1]}")
plt.show()
