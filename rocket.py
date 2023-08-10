import random
import numpy as np


class Hopper:
    # Physical Properties of the hopper
    m = 3  # Hopper weight in [kg]
    MaxThrust = 50  # Maximum possible thrust in [N]
    p_amb = 101300  # Ambient pressure in [Pa]

    # Properties of working fluid Nitrogen - [N2] --> Assuming ideal gas behaviour
    gamma = 1.4  # isentropic exponent [-]
    p0 = 1100000  # Inlet pressure of Valve
    T1 = 298  # Temperature after valve [K]
    R_s = 296.8  # specific gas constant [J/kg*K]

    # Properties of the Nozzle
    d_th = 0.011  # Throat diameter in [m]
    epsilon = 1.35  # Expansion ratio
    A_th = np.pi * 0.25 * d_th**2  # Throat Area in [m^2]
    A_e = epsilon * A_th  # Exit Nozzle area in [m^2]

    def m_dot(self, p1):
        phi = np.sqrt(self.gamma * (2 / (self.gamma+1))**((self.gamma+1)/(self.gamma-1)))
        m_dot = (p1 * self.A_th / np.sqrt(self.R_s * self.T1)) * phi
        return m_dot

    def p_e(self, p1):

        p_e_new = 10000
        p_e_old = 0

        while abs(p_e_old - p_e_new) > 0.002:
            p_e_old = p_e_new
            phi1 = np.sqrt((self.gamma - 1) * 0.5) * (2 / (self.gamma+1))**((self.gamma+1)/(self.gamma-1))
            phi2 = self.epsilon * np.sqrt(1 - (p_e_old/p1)**((self.gamma-1)/self.gamma))
            p_e_new = p1 * (phi1 / phi2)**self.gamma

        p_e = p_e_new
        return p_e

    def v_e(self, p1, p_e):
        v_e = np.sqrt(2*self.gamma/(self.gamma-1) * self.R_s * self.T1 * (1 - (p_e/p1)**(self.gamma-1)/self.gamma))
        return v_e

    def thrust(self, p1):
        p_e = self.p_e(p1)
        thrust = self.m_dot(p1) * self.v_e(p1, p_e) + (p_e - self.p_amb) * self.A_e
        return thrust


# rocket = Hopper()
#
# print(rocket.thrust(17 * 10000 + 100000))


class HopperEnv:
    # Physical Properties of the Hopper & the Environment
    g = 9.81  # Gravitational acceleration of the Earth [m/s^2]
    p_amb = 101300  # Ambient pressure in [Pa]

    # Time and Step variable
    duration = 20
    t0 = 0
    time_steps_per_second = 10  # Sample time of the sensors
    N = duration * time_steps_per_second  # Number of Time-Steps
    h = duration / N  # Step size

    # Counters and Variables
    episode_step = 0

    # RL related constants for the environment
    BOUNDARY_PENALTY = 10
    sigma_1 = 2

    # Action Space and Observation Space Sizes
    action_space = 100
    observation_space = 3

    # Initial state
    x0 = 0
    v0 = 0

    # Random target state (altitude)
    xt = random.uniform(0, 8)

    # Use Hopper Equations in Hopper Environment
    rocket = Hopper()

    def f(self, t, y, action):

        x = y[0]
        v = y[1]

        thrust = self.rocket.thrust(action * 10000 + 100000)

        cd = 0.6
        A = 0.1 * 0.3
        rho = 1.225
        F_aero = 0.5 * cd * rho * A * v * abs(v)

        x_dot = v
        v_dot = thrust / Hopper.m - self.g - F_aero / Hopper.m

        return np.array([x_dot, v_dot])

    def ode45_step(self, f, y, t, h, action):
        # runge kutta 4th order explicit
        tk_05 = t + 0.5 * h
        yk_025 = y + 0.5 * h * f(t, y, action)
        yk_05 = y + 0.5 * h * f(tk_05, yk_025, action)
        yk_075 = y + h * f(tk_05, yk_05, action)

        return y + h / 6 * (
                f(t, y, action) + 2 * f(tk_05, yk_025, action) + 2 * f(tk_05, yk_05, action) + f(t + h, yk_075, action))

    def reset(self):
        # New random altitude goal
        self.xt = random.uniform(0, 8)

        # Calculate the new initial state vector
        s0 = [self.x0 - self.xt, self.v0]
        s_init = s0
        y = s_init
        action = 0

        # Reset values
        self.episode_step = 0

        observation = np.concatenate((y, np.array([action])))
        return observation

    def step(self, action, state):
        # rename the state as y
        y = np.array([state[0], state[1]])
        # What does it take to make a step
        self.episode_step += 1
        t = self.t0 + self.episode_step * self.h
        y = self.ode45_step(self.f, y, t, self.h, action)

        if y[0] + self.xt < 0:
            y[0] = self.x0 - self.xt
            y[1] = 0
        elif y[0] + self.xt > 10:
            y[0] = 10 - self.xt
            y[1] = 0

        # By giving the action in the state we give the Network the current valve position
        y = np.concatenate((y, np.array([action])))

        # Define the reward
        if abs(y[0]) < self.sigma_1:
            rew1 = 1 - np.sqrt(abs(y[0])/self.sigma_1)
        else:
            rew1 = 0

        if y[0] + self.xt == 10 or y[0] + self.xt == 0:
            penalty = -self.BOUNDARY_PENALTY
        else:
            penalty = 0

        reward = rew1 + penalty

        done = False
        if self.episode_step >= 200:
            done = True

        return y, reward, done
