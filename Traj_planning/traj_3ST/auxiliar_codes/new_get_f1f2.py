import casadi as ca
from Traj_planning.traj_3ST.auxiliar_codes.new_get_gamma import *

def get_f1f2(x_dot_dot, y_dot_dot, x_3dot, y_3dot, x_4dot, y_4dot, params):
    g = params["g"]
    m = params["m"]
    J_z = params["J_z"]
    l_tvc = params["l_tvc"]
    
    e1bx = x_dot_dot / ca.sqrt(x_dot_dot**2 + (y_dot_dot + g) ** 2)
    e1by = (y_dot_dot + g) / ca.sqrt(x_dot_dot**2 + (y_dot_dot + g) ** 2)
    
    gamma_dot = (y_3dot * x_dot_dot - x_3dot * (y_dot_dot + g)) / (
        x_dot_dot**2 + (y_dot_dot + g) ** 2)
    
    gamma_dot_dot = get_gamma_2dot(x_dot_dot, y_dot_dot, x_3dot, y_3dot, x_4dot, y_4dot, params)

    f1 = m * (x_dot_dot * e1bx + (y_dot_dot + g) * e1by) + gamma_dot**2 * J_z / l_tvc
    f2 = -J_z * gamma_dot_dot / l_tvc

    return f1, f2

def get_f1f2_dot(x_dot_dot, y_dot_dot, x_3dot, y_3dot, x_4dot, y_4dot, x_5dot, y_5dot, params):
    g = params["g"]
    m = params["m"]
    J_z = params["J_z"]
    l_tvc = params["l_tvc"]
    
    e1bx = x_dot_dot / ca.sqrt(x_dot_dot**2 + (y_dot_dot + g) ** 2)
    e1by = (y_dot_dot + g) / ca.sqrt(x_dot_dot**2 + (y_dot_dot + g) ** 2)
    
    gamma_dot = (y_3dot * x_dot_dot - x_3dot * (y_dot_dot + g)) / (x_dot_dot**2 + (y_dot_dot + g) ** 2)
    gamma_dot_dot = get_gamma_2dot(x_dot_dot, y_dot_dot, x_3dot, y_3dot, x_4dot, y_4dot, params)
    gamma_3dot = get_gamma_3dot(x_dot_dot, y_dot_dot, x_3dot, y_3dot, x_4dot, y_4dot, x_5dot, y_5dot, params)
    
    f1_dot = m * (x_3dot * e1bx - x_dot_dot * gamma_dot * e1by + y_3dot * e1by + (y_dot_dot + g) * gamma_dot * e1bx) + 2 * gamma_dot * gamma_dot_dot * J_z / l_tvc

    f2_dot = -J_z * gamma_3dot / l_tvc
    
    return f1_dot, f2_dot
