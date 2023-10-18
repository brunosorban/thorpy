import casadi as ca
from Traj_planning.auxiliar_codes.compute_omega import compute_omega
from Traj_planning.auxiliar_codes.compute_omega_dot import compute_omega_dot
from Traj_planning.auxiliar_codes.compute_omega_dot_dot import compute_omega_dot_dot


def compute_f1f2f3(
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
    """This function computes the f1, f2 and f3 forces, which are used in the differential flatness approach.
    The f1 and f2 forces are always perpendicular to the z axis, the f3 force is always parallel to the z axis. 

    Args:
        x_dot_dot (_type_): acceleration in the x direction
        y_dot_dot (_type_): acceleration in the y direction
        z_dot_dot (_type_): acceleration in the z direction
        x_3dot (_type_): jerk in the x direction
        y_3dot (_type_): jerk in the y direction
        z_3dot (_type_): jerk in the z direction
        x_4dot (_type_): snap in the x direction
        y_4dot (_type_): snap in the y direction
        z_4dot (_type_): snap in the z direction
        x_5dot (_type_): crackle in the x direction
        y_5dot (_type_): crackle in the y direction
        z_5dot (_type_): crackle in the z direction
        params (_type_): dictionary containing the parameters of the rocket. Currently mass (m), inertia in x and y (J), 
        distance between the thruster and the center of mass (l_tvc) and gravity (g) are used.

    Returns:
        f1: force perpendicular to the z axis in the x direction
        f2: force perpendicular to the z axis in the y direction
        f3: force parallel to the z axis
        f1_dot: time derivative of f1
        f2_dot: time derivative of f2
        f3_dot: time derivative of f3
    """
    m = params["m"]
    J = params["J"]
    l_tvc = params["l_tvc"]
    g = params["g"]

    t = ca.vertcat(x_dot_dot, y_dot_dot, z_dot_dot + g)

    zb = t / ca.norm_2(t)
    xc = ca.vertcat(1, 0, 0)  # heading always in the x direction
    yb = ca.cross(zb, xc) / ca.norm_2(ca.cross(zb, xc))
    xb = ca.cross(yb, zb)

    omega = compute_omega(x_dot_dot, y_dot_dot, z_dot_dot, x_3dot, y_3dot, z_3dot, g)
    omega_dot = compute_omega_dot(
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
    f1 = -J / l_tvc * omega_dot_2
    f2 = J / l_tvc * omega_dot_1

    f1_dot = -J / l_tvc * omega_dot_dot_2
    f2_dot = J / l_tvc * omega_dot_dot_1
    f3_dot = m * (
        x_dot_dot * x_3dot + y_dot_dot * y_3dot + (z_dot_dot + g) * z_3dot
    ) / ca.norm_2(t) + 2 * J / l_tvc * (omega_1 * omega_dot_1 + omega_2 * omega_dot_2)

    return f1, f2, f3, f1_dot, f2_dot, f3_dot
