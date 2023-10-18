def RK4(fun, x, u, dt):
    k1 = fun(x, u)
    k2 = fun(x + dt / 2 * k1, u)
    k3 = fun(x + dt / 2 * k2, u)
    k4 = fun(x + dt * k3, u)
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next
