import casadi as ca
from Traj_planning.auxiliar_codes.compute_omega import compute_omega
from Traj_planning.auxiliar_codes.compute_omega_dot import compute_omega_dot
from Traj_planning.auxiliar_codes.compute_omega_dot_dot import compute_omega_dot_dot


def compute_f1f2f3(
    time,
    x_dot_dot,
    y_dot_dot,
    z_dot_dot,
    x_3dot,
    y_3dot,
    z_3dot,
    x_4dot,
    y_4dot,
    z_4dot,
    x_5dot,
    y_5dot,
    z_5dot,
    params,
):
    m = params["m"]
    I = params["I"]
    J = params["J"]
    l_tvc = params["l_tvc"]
    g = params["g"]
    
    t = ca.vertcat(x_dot_dot, y_dot_dot, z_dot_dot + g)
    # r_3dot = ca.vertcat(x_3dot, y_3dot, z_3dot)
    # r_4dot = ca.vertcat(x_4dot, y_4dot, z_4dot)
    
    zb = t / ca.norm_2(t)
    xc = ca.vertcat(1, 0 ,0) # heading always in the x direction
    yb = ca.cross(zb, xc) / ca.norm_2(ca.cross(zb, xc))
    xb = ca.cross(yb, zb) / ca.norm_2(ca.cross(yb, zb))
    
    omega =compute_omega(x_dot_dot, y_dot_dot, z_dot_dot, x_3dot, y_3dot, z_3dot, g)
    omega_dot =compute_omega_dot(
        x_dot_dot,
        y_dot_dot,
        z_dot_dot,
        x_3dot,
        y_3dot,
        z_3dot,
        x_4dot,
        y_4dot,
        z_4dot,
        g,
    )
    omega_dot_dot = compute_omega_dot_dot(
        x_dot_dot,
        y_dot_dot,
        z_dot_dot,
        x_3dot,
        y_3dot,
        z_3dot,
        x_4dot,
        y_4dot,
        z_4dot,
        x_5dot,
        y_5dot,
        z_5dot,
        g,
    )
    
    omega_1 = ca.dot(omega_dot, xb)
    omega_2 = ca.dot(omega_dot, yb)
    
    omega_dot_1 = ca.dot(omega_dot_dot, xb)
    omega_dot_2 = ca.dot(omega_dot_dot, yb)
    
    omega_dot_dot_1 = ca.dot(omega_dot_dot, xb)
    omega_dot_dot_2 = ca.dot(omega_dot_dot, yb)
    
    f3 = m * ca.norm_2(t) + J / l_tvc * ca.power(omega, 2)
    f1 = - J / l_tvc * omega_dot_2
    f2 = J / l_tvc * omega_dot_1
    
    f1_dot = - J / l_tvc * omega_dot_dot_2
    f2_dot = J / l_tvc * omega_dot_dot_1
    f3_dot = m * (x_dot_dot * x_3dot + y_dot_dot * y_3dot + (z_dot_dot + g) * z_3dot) / ca.norm_2(t) + 2 * J / l_tvc * (omega_1 * omega_dot_1 + omega_2 * omega_dot_2)
    
    return f1, f2, f3, f1_dot, f2_dot, f3_dot
