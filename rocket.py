import random
import numpy as np

class HopperEnv:
    # Physical Properties of the Hopper & the Environment
    m = 3
    g = 9.81
    MaxThrust = 50  # Maximum possible thrust in N

    # Time and Step variable
    duration = 20
    t0 = 0
    time_steps_per_second = 10  # Sample time of the sensors
    N = duration * time_steps_per_second  # Number of Time-Steps
    h = duration / N  # Step size

    # Counters and Variables
    episode_step = 0
    margin = 0.05  # Allowed stationary offset of 5cm

    # RL related constants for the environment
    # MOVE_PENALTY = 0
    BOUNDARY_PENALTY = 10
    sigma_1 = 2
    # HEIGHT_REWARD = 10
    # HEIGHT_STEP_REWARD = 2

    # Action Space and Observation Space Sizes
    action_space = 100
    observation_space = 3

    # Initial and target state
    x0 = 0
    v0 = 0

    # New random altitude goal
    xt = random.uniform(0, 8)

    def thrust_eqn(self, action):
        thrust = action/100 * self.MaxThrust
        return thrust

    def f(self, t, y, action):

        x = y[0]
        v = y[1]

        thrust = HopperEnv.thrust_eqn(self, action)

        cd = 0.6
        A = 0.1 * 0.3
        rho = 1.225
        F_aero = 0.5 * cd * rho * A * v * abs(v)

        x_dot = v
        v_dot = thrust / self.m - self.g - F_aero / self.m

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

        y = np.concatenate((y, np.array([action])))  # By giving the action in the state we give the Network the current valve position

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

        # if self.xt + y[0] < 0:
        #     reward = -self.BOUNDARY_PENALTY
        # elif self.xt + y[0] > 10:
        #     reward = -self.BOUNDARY_PENALTY
        # elif abs(y[0]) < self.margin:
        #     reward = self.HEIGHT_REWARD
        # elif abs(y[0]) < 0.5:
        #     reward = self.HEIGHT_STEP_REWARD
        # else:
        #     reward = -self.MOVE_PENALTY

        done = False
        if self.episode_step >= 200:
            done = True

        return y, reward, done

    # def render(self):
    #     img = self.get_image()
    #     img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
    #     cv2.imshow("image", np.array(img))  # show it!
    #     cv2.waitKey(1)
