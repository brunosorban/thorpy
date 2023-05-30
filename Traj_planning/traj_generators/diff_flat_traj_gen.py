import numpy as np
import matplotlib.pyplot as plt

def diff_flat_traj(x, y, z, total_time, constraints=False, g=9.81, optimize=False):
    gain = 1.5
    t = np.linspace(0, total_time, len(x))
    x_dot = np.gradient(x, t)
    x_dot_dot = np.gradient(x_dot, t)
    
    y_dot = np.gradient(y, t)
    y_dot_dot = np.gradient(y_dot, t)
    
    z_dot = np.gradient(z, t)
    z_dot_dot = np.gradient(z_dot, t)

    gamma = np.arctan2(y_dot_dot + g, x_dot_dot)
    gamma_dot = np.gradient(gamma, t)
    gamma_dot_dot = np.gradient(gamma_dot, t)
    
    e1bx = np.cos(gamma)
    e1by = np.sin(gamma)
    e2bx = -np.sin(gamma)
    e2by = np.cos(gamma)
    
    if optimize:
        max_ax = constraints["max_ax"]
        min_ax = constraints["min_ax"]
        max_ay = constraints["max_ay"]
        min_ay = constraints["min_ay"]
        optimizing = True
        i = 1
        
        print("Optimizing trajectory...")
        print("max_ax: ", max_ax)
        print("min_ax: ", min_ax)
        print("max_ay: ", max_ay)
        print("min_ay: ", min_ay)
        print()
        print("ax0: ", x_dot_dot[0])
        print("ay0: ", y_dot_dot[0])
        
        while optimizing:
            dt = t[i+1] - t[i]
            
            # check if bounds were violated
            if x_dot_dot[i] > max_ax:
                temp = dt * gain * (x_dot_dot[i] - max_ax) / max_ax
                if temp < 1e-2: temp = 1e-2
                t[i+1:] += temp
                
            elif x_dot_dot[i] < min_ax:
                temp = dt * gain * (x_dot_dot[i] - min_ax) / min_ax
                if temp < 1e-2: temp = 1e-2
                t[i+1:] += temp
                
            elif y_dot_dot[i] > max_ay:
                temp = dt * gain * (y_dot_dot[i] - max_ay) / max_ay
                if temp < 1e-2: temp = 1e-2
                t[i+1:] += temp
                
            elif y_dot_dot[i] < min_ay:
                temp = dt * gain * (y_dot_dot[i] - min_ay) / min_ay
                if temp < 1e-2: temp = 1e-2
                t[i+1:] += temp
                
            # # accelerate if too slow
            # elif 0 < x_dot_dot[i] < 0.5 * max_ax:
            #     temp = dt * gain * (max_ax - x_dot_dot[i]) / max_ax
            #     if temp < 1e-2: temp = 1e-2
            #     t[i+1:] -= temp
                
            # elif 0.5 * min_ax < x_dot_dot[i] < 0:
            #     temp = dt * gain * (min_ax - x_dot_dot[i]) / min_ax
            #     if temp < 1e-2: temp = 1e-2
            #     t[i+1:] -= temp
                
            # elif 0 < y_dot_dot[i] < 0.5 * max_ay:
            #     temp = dt * gain * (max_ay - y_dot_dot[i]) / max_ay
            #     if temp < 1e-2: temp = 1e-2
            #     t[i+1:] -= temp
                
            # elif 0.5 * min_ay < y_dot_dot[i] < 0:
            #     temp = dt * gain * (min_ay - y_dot_dot[i]) / min_ay
            #     if temp < 1e-2: temp = 1e-2
            #     t[i+1:] -= temp
                
            else:
                i += 1
            
            if i >= len(t) -1:
                optimizing = False
            
            x_dot = np.gradient(x, t)
            x_dot_dot = np.gradient(x_dot, t)
            y_dot = np.gradient(y, t)
            y_dot_dot = np.gradient(y_dot, t)
            z_dot = np.gradient(z, t)
            z_dot_dot = np.gradient(z_dot, t)
            gamma = np.arctan2(y_dot_dot + g, x_dot_dot)
            gamma_dot = np.gradient(gamma, t)
            gamma_dot_dot = np.gradient(gamma_dot, t)
            e1bx = np.cos(gamma)
            e1by = np.sin(gamma)
            e2bx = -np.sin(gamma)
            e2by = np.cos(gamma)
            
            # print("i: ", i)
            # print("t: ", t[i])
            # print("ax: ", x_dot_dot[i])
            # print("ay: ", y_dot_dot[i])
            
            # input("Press Enter to continue...")
                

    return (
        t,
        x,
        x_dot,
        x_dot_dot,
        y,
        y_dot,
        y_dot_dot,
        z,
        z_dot,
        z_dot_dot,
        e1bx,
        e1by,
        e2bx,
        e2by,
        gamma_dot,
        gamma_dot_dot,)

def get_traj_params(states, constraints, plot=False):
    x = states["x"]
    y = states["y"]
    z = states["z"]
    vx = states["vx"]
    vy = states["vy"]
    vz = states["vz"]
    ax = states["ax"]
    ay = states["ay"]
    az = states["az"]
    e1bx = states["e1bx"]
    e1by = states["e1by"]
    e2bx = states["e2bx"]
    e2by = states["e2by"]
    gamma_dot = states["gamma_dot"]
    gamma_dot_dot = states["gamma_dot_dot"]
    t = states["t"]
    
    gamma = np.arctan2(e1by, e1bx)
    
    trajectory = {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "ax": ax,
        "ay": ay,
        "az": az,
        "e1bx": e1bx,
        "e1by": e1by,
        "e2bx": e2bx,
        "e2by": e2by,
        "gamma": gamma,
        "gamma_dot": gamma_dot,
    }

    if plot:
        plot_trajectory(t, x, vx, ax, y, vy, ay, z, vz, az, gamma, gamma_dot, gamma_dot_dot)

    return trajectory

def plot_trajectory(t, x, vx, ax, y, vy, ay, z, vz, az, gamma, gamma_dot, gamma_dot_dot):
    e1bx = np.cos(gamma)
    e1by = np.sin(gamma)
    e2bx = -np.sin(gamma)
    e2by = np.cos(gamma)
    last_t = t[0]
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs[0, 0].plot(x, y, label="trajectory")
    axs[0, 0].scatter([-100, 100], [0, 0], s=1e-3)
    # axs[0, 0].plot(states["x"], states["y"], "o", label="target points")
    axs[0, 0].set_xlabel("Horizontal Position (m)")
    axs[0, 0].set_ylabel("Vertical Position (m)")
    axs[0, 0].set_title("Trajectory")
    axs[0, 0].set_aspect("equal")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # make the arrows
    for k in range(0, len(t) - 1):
        temp = t[t >= t[k]]

        for i in range(len(temp)):
            if (
                temp[i] - last_t >= 1 and temp[i] < t[k + 1]
            ):  # Plot frame every 10 seconds
                ind = np.searchsorted(t, temp[i])
                origin_x = x[ind]
                origin_y = y[ind]
                e1_x = e1bx[ind]
                e1_y = e1by[ind]
                e2_x = e2bx[ind]
                e2_y = e2by[ind]
                axs[0, 0].quiver(origin_x, origin_y, e1_x, e1_y, scale=10, color="red")
                axs[0, 0].quiver(origin_x, origin_y, e2_x, e2_y, scale=10, color="blue")
                print(
                    "At time: {:.1f}, x: {:.1f}, y: {:.1f} e1x: {:.1f}, e1y: {:.1f}, e2x: {:.1f}, e2y: {:.1f}".format(
                        temp[i], origin_x, origin_y, e1_x, e1_y, e2_x, e2_y
                    )
                )
                last_t = temp[i]

    axs[1, 0].plot(t, x, label="x")
    axs[1, 0].plot(t, y, label="y")
    # axs[1, 0].plot(t, states["x"], "o", label="$x_{ref}$")
    # axs[1, 0].plot(t, states["y"], "o", label="$y_{ref}$")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Position (m)")
    axs[1, 0].set_title("Position vs Time")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[2, 0].plot(t, np.rad2deg(gamma), label="$\\gamma$")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Angle (deg)")
    axs[2, 0].set_title("Angle vs Time")
    axs[2, 0].legend()
    axs[2, 0].grid()

    axs[0, 1].plot(t, vx, label="vx")
    axs[0, 1].plot(t, vy, label="vy")
    # axs[0, 1].plot(t, states["vx"], "o", label="$v_{x_{ref}}$")
    # axs[0, 1].plot(t, states["vy"], "o", label="$v_{y_{ref}}$")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].set_title("Velocity vs Time")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 1].plot(t, ax, label="ax")
    axs[1, 1].plot(t, ay, label="ay")
    # axs[1, 1].plot(t, states["ax"], "o", label="$a_{x_{ref}}$")
    # axs[1, 1].plot(t, states["ay"], "o", label="$a_{y_{ref}}$")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Acceleration (m/s^2)")
    axs[1, 1].set_title("Acceleration vs Time")
    axs[1, 1].legend()
    axs[1, 1].grid()

    # axs[2, 1].plot(t, np.rad2deg(gamma_dot), label="$\\dot{\\gamma}$")
    axs[2, 1].plot(t, e1bx, label="$e_{1bx}$")
    axs[2, 1].plot(t, e1by, label="$e_{1by}$")
    axs[2, 1].plot(t, e2bx, label="$e_{2bx}$")
    axs[2, 1].plot(t, e2by, label="$e_{2by}$")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Frame parameters (-)")
    axs[2, 1].set_title("Frame parameters vs Time")
    axs[2, 1].legend()
    axs[2, 1].grid()

    plt.show()