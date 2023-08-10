import numpy as np
import matplotlib.pyplot as plt
import rocket

# Test of the ODE and step() + reset() fcn
duration = 20
N = 200

env = environment.HopperEnv()

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
