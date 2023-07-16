def get_pos(coefs, t):
    order = coefs.shape[0]
    pos = 0

    for i in range(order):
        pos += coefs[i] * t**i
    return pos


def get_vel(coefs, t):
    order = coefs.shape[0]
    vel = 0

    if order == 0:
        return vel

    for i in range(order):
        if i > 0:  # discard the constant term
            vel += i * coefs[i] * t ** (i - 1)
    return vel


def get_acc(coefs, t):
    order = coefs.shape[0]
    acc = 0

    if order <= 1:
        return acc

    for i in range(order):
        if i > 1:  # discard the constant and linear terms
            acc += i * (i - 1) * coefs[i] * t ** (i - 2)
    return acc


def get_jerk(coefs, t):
    order = coefs.shape[0]
    jerk = 0

    if order <= 2:
        return jerk

    for i in range(order):
        if i > 2:  # discard the constant, linear and quadratic terms
            jerk += i * (i - 1) * (i - 2) * coefs[i] * t ** (i - 3)
    return jerk


def get_snap(coefs, t):
    order = coefs.shape[0]
    snap = 0

    if order <= 3:
        return snap

    for i in range(order):
        if i > 3:  # discard the constant, linear, quadratic and cubic terms
            snap += i * (i - 1) * (i - 2) * (i - 3) * coefs[i] * t ** (i - 4)
    return snap


def get_crackle(coefs, t):
    order = coefs.shape[0]
    crackle = 0

    if order <= 4:
        return crackle

    for i in range(order):
        if i > 4:  # discard the constant, linear, quadratic, cubic and quartic terms
            crackle += (
                i * (i - 1) * (i - 2) * (i - 3) * (i - 4) * coefs[i] * t ** (i - 5)
            )
    return crackle


# def get_pop(coefs, t):
#     order = coefs.shape[0]
#     pop = 0

#     if order <= 5:
#         return pop

#     for i in range(order):
#         if i > 5: # discard the constant, linear, quadratic, cubic, quartic and quintic terms
#             pop += i * (i-1) * (i-2) * (i-3) * (i-4) * (i-5) * coefs[i] * t**(i-6)
#     return pop
