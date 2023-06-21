# def get_pos(coefs, t):
#     return (
#         coefs[0] * t**7
#         + coefs[1] * t**6
#         + coefs[2] * t**5
#         + coefs[3] * t**4
#         + coefs[4] * t**3
#         + coefs[5] * t**2
#         + coefs[6] * t
#         + coefs[7]
#     )


# def get_vel(coefs, t):
#     return (
#         7 * coefs[0] * t**6
#         + 6 * coefs[1] * t**5
#         + 5 * coefs[2] * t**4
#         + 4 * coefs[3] * t**3
#         + 3 * coefs[4] * t**2
#         + 2 * coefs[5] * t
#         + coefs[6]
#     )


# def get_acc(coefs, t):
#     return (
#         42 * coefs[0] * t**5
#         + 30 * coefs[1] * t**4
#         + 20 * coefs[2] * t**3
#         + 12 * coefs[3] * t**2
#         + 6 * coefs[4] * t
#         + 2 * coefs[5]
#     )


# def get_jerk(coefs, t):
#     return (
#         210 * coefs[0] * t**4
#         + 120 * coefs[1] * t**3
#         + 60 * coefs[2] * t**2
#         + 24 * coefs[3] * t
#         + 6 * coefs[4]
#     )


# def get_snap(coefs, t):
#     return (
#         840 * coefs[0] * t**3
#         + 360 * coefs[1] * t**2
#         + 120 * coefs[2] * t
#         + 24 * coefs[3]
#     )


# def get_crackle(coefs, t):
#     return 2520 * coefs[0] * t**2 + 720 * coefs[1] * t + 120 * coefs[2]


# def get_pop(coefs, t):
#     return 5040 * coefs[0] * t + 720 * coefs[1]


# def get_pos(coefs, t):
#     return (
#         coefs[0] * t**8
#         + coefs[1] * t**7
#         + coefs[2] * t**6
#         + coefs[3] * t**5
#         + coefs[4] * t**4
#         + coefs[5] * t**3
#         + coefs[6] * t**2
#         + coefs[7] * t
#         + coefs[8]
#     )


# def get_vel(coefs, t):
#     return (
#         8 * coefs[0] * t**7
#         + 7 * coefs[1] * t**6
#         + 6 * coefs[2] * t**5
#         + 5 * coefs[3] * t**4
#         + 4 * coefs[4] * t**3
#         + 3 * coefs[5] * t**2
#         + 2 * coefs[6] * t
#         + coefs[7]
#     )


# def get_acc(coefs, t):
#     return (
#         56 * coefs[0] * t**6
#         + 42 * coefs[1] * t**5
#         + 30 * coefs[2] * t**4
#         + 20 * coefs[3] * t**3
#         + 12 * coefs[4] * t**2
#         + 6 * coefs[5] * t
#         + 2 * coefs[6]
#     )


# def get_jerk(coefs, t):
#     return (
#         336 * coefs[0] * t**5
#         + 210 * coefs[1] * t**4
#         + 120 * coefs[2] * t**3
#         + 60 * coefs[3] * t**2
#         + 24 * coefs[4] * t
#         + 6 * coefs[5]
#     )


# def get_snap(coefs, t):
#     return (
#         1680 * coefs[0] * t**4
#         + 840 * coefs[1] * t**3
#         + 360 * coefs[2] * t**2
#         + 120 * coefs[3] * t
#         + 24 * coefs[4]
#     )


# def get_crackle(coefs, t):
#     return (
#         6720 * coefs[0] * t**3
#         + 2520 * coefs[1] * t**2
#         + 720 * coefs[2] * t
#         + 120 * coefs[3]
#     )


# def get_pop(coefs, t):
#     return 20160 * coefs[0] * t**2 + 5040 * coefs[1] * t + 720 * coefs[2]


# def get_pos(coefs, t):
#     return (
#         coefs[0] * t ** 9
#         + coefs[1] * t ** 8
#         + coefs[2] * t ** 7
#         + coefs[3] * t ** 6
#         + coefs[4] * t ** 5
#         + coefs[5] * t ** 4
#         + coefs[6] * t ** 3
#         + coefs[7] * t ** 2
#         + coefs[8] * t
#         + coefs[9]
#     )

# def get_vel(coefs, t):
#     return (
#         9 * coefs[0] * t ** 8
#         + 8 * coefs[1] * t ** 7
#         + 7 * coefs[2] * t ** 6
#         + 6 * coefs[3] * t ** 5
#         + 5 * coefs[4] * t ** 4
#         + 4 * coefs[5] * t ** 3
#         + 3 * coefs[6] * t ** 2
#         + 2 * coefs[7] * t
#         + coefs[8]
#     )

# def get_acc(coefs, t):
#     return (
#         72 * coefs[0] * t ** 7
#         + 56 * coefs[1] * t ** 6
#         + 42 * coefs[2] * t ** 5
#         + 30 * coefs[3] * t ** 4
#         + 20 * coefs[4] * t ** 3
#         + 12 * coefs[5] * t ** 2
#         + 6 * coefs[6] * t
#         + 2 * coefs[7]
#     )

# def get_jerk(coefs, t):
#     return (
#         504 * coefs[0] * t ** 6
#         + 336 * coefs[1] * t ** 5
#         + 210 * coefs[2] * t ** 4
#         + 120 * coefs[3] * t ** 3
#         + 60 * coefs[4] * t ** 2
#         + 24 * coefs[5] * t
#         + 6 * coefs[6]
#     )

# def get_snap(coefs, t):
#     return (
#         3024 * coefs[0] * t ** 5
#         + 1680 * coefs[1] * t ** 4
#         + 840 * coefs[2] * t ** 3
#         + 360 * coefs[3] * t ** 2
#         + 120 * coefs[4] * t
#         + 24 * coefs[5]
#     )

# def get_crackle(coefs, t):
#     return (
#         15120 * coefs[0] * t ** 4
#         + 6720 * coefs[1] * t ** 3
#         + 2520 * coefs[2] * t ** 2
#         + 720 * coefs[3] * t
#         + 120 * coefs[4]
#     )

# def get_pop(coefs, t):
#     return (
#         60480 * coefs[0] * t ** 3
#         + 20160 * coefs[1] * t ** 2
#         + 5040 * coefs[2] * t
#         + 720 * coefs[3]
#     )

########################################################## 12th order
def get_pos(coefs, t):
    return (
        coefs[0] * t ** 12
        + coefs[1] * t ** 11
        + coefs[2] * t ** 10
        + coefs[3] * t ** 9
        + coefs[4] * t ** 8
        + coefs[5] * t ** 7
        + coefs[6] * t ** 6
        + coefs[7] * t ** 5
        + coefs[8] * t ** 4
        + coefs[9] * t ** 3
        + coefs[10] * t ** 2
        + coefs[11] * t
        + coefs[12]
    )
    
def get_vel(coefs, t):
    return (
        12 * coefs[0] * t ** 11
        + 11 * coefs[1] * t ** 10
        + 10 * coefs[2] * t ** 9
        + 9 * coefs[3] * t ** 8
        + 8 * coefs[4] * t ** 7
        + 7 * coefs[5] * t ** 6
        + 6 * coefs[6] * t ** 5
        + 5 * coefs[7] * t ** 4
        + 4 * coefs[8] * t ** 3
        + 3 * coefs[9] * t ** 2
        + 2 * coefs[10] * t
        + coefs[11]
    )
    
def get_acc(coefs, t):
    return (
        132 * coefs[0] * t ** 10
        + 110 * coefs[1] * t ** 9
        + 90 * coefs[2] * t ** 8
        + 72 * coefs[3] * t ** 7
        + 56 * coefs[4] * t ** 6
        + 42 * coefs[5] * t ** 5
        + 30 * coefs[6] * t ** 4
        + 20 * coefs[7] * t ** 3
        + 12 * coefs[8] * t ** 2
        + 6 * coefs[9] * t
        + 2 * coefs[10]
    )
    
def get_jerk(coefs, t):
    return (
        1320 * coefs[0] * t ** 9
        + 990 * coefs[1] * t ** 8
        + 720 * coefs[2] * t ** 7
        + 504 * coefs[3] * t ** 6
        + 336 * coefs[4] * t ** 5
        + 210 * coefs[5] * t ** 4
        + 120 * coefs[6] * t ** 3
        + 60 * coefs[7] * t ** 2
        + 24 * coefs[8] * t
        + 6 * coefs[9]
    )
    
def get_snap(coefs, t):
    return (
        11880 * coefs[0] * t ** 8
        + 7920 * coefs[1] * t ** 7
        + 5040 * coefs[2] * t ** 6
        + 3024 * coefs[3] * t ** 5
        + 1680 * coefs[4] * t ** 4
        + 840 * coefs[5] * t ** 3
        + 360 * coefs[6] * t ** 2
        + 120 * coefs[7] * t
        + 24 * coefs[8]
    )
    
def get_crackle(coefs, t):
    return (
        95040 * coefs[0] * t ** 7
        + 55440 * coefs[1] * t ** 6
        + 30240 * coefs[2] * t ** 5
        + 15120 * coefs[3] * t ** 4
        + 6720 * coefs[4] * t ** 3
        + 2520 * coefs[5] * t ** 2
        + 720 * coefs[6] * t
        + 120 * coefs[7]
    )
    
def get_pop(coefs, t):
    return (
        665280 * coefs[0] * t ** 6
        + 332640 * coefs[1] * t ** 5
        + 151200 * coefs[2] * t ** 4
        + 60480 * coefs[3] * t ** 3
        + 20160 * coefs[4] * t ** 2
        + 5040 * coefs[5] * t
        + 720 * coefs[6]
    )