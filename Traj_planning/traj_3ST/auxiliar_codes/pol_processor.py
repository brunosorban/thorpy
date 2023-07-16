def pos_processor(coeffs, time_interval, cur_time):
    """
    This function returns the value of the polinomial at the given time.

    Parameters
    ----------
    coeffs : list
        The coefficients of the polinomial.
    time_interval : list
        The time interval of the polinomial.
    cur_time : float
        The time list at which the polinomial should be evaluated. Should be an
        increasing list.

    Returns
    -------
    float
        The value of the polinomial at the given time.
    """
    if cur_time[0] < time_interval[0] or cur_time[1] > time_interval[1]:
        raise ValueError(
            "The given time is not in the time interval of the polinomial."
        )

    scaled_time = (cur_time - time_interval[0]) / (time_interval[1] - time_interval[0])

    # define the polinomial
    pos = 0
    for i in range(len(coeffs)):
        pos += coeffs[i] * (scaled_time**i)

    return pos


def vel_processor(coeffs, time_interval, cur_time):
    """
    This function returns the value of the first derivative of the polinomial at the given time.

    Parameters
    ----------
    coeffs : list
        The coefficients of the polinomial.
    time_interval : list
        The time interval of the polinomial.
    cur_time : float
        The time at which the polinomial should be evaluated.

    Returns
    -------
    float
        The value of the first derivative of the polinomial at the given time.
    """
    if cur_time[0] < time_interval[0] or cur_time[1] > time_interval[1]:
        raise ValueError(
            "The given time is not in the time interval of the polinomial."
        )

    dt = time_interval[1] - time_interval[0]
    scaled_time = (cur_time - time_interval[0]) / dt

    # define the polinomial
    vel = 0
    for i in range(1, len(coeffs)):
        vel += i * coeffs[i] * (scaled_time ** (i - 1)) / dt

    return vel


def acc_processor(coeffs, time_interval, cur_time):
    """
    This function returns the value of the second derivative of the polinomial at the given time.

    Parameters
    ----------
    coeffs : list
        The coefficients of the polinomial.
    time_interval : list
        The time interval of the polinomial.
    cur_time : float
        The time at which the polinomial should be evaluated.

    Returns
    -------
    float
        The value of the second derivative of the polinomial at the given time.
    """
    if cur_time[0] < time_interval[0] or cur_time[1] > time_interval[1]:
        raise ValueError(
            "The given time is not in the time interval of the polinomial."
        )

    dt = time_interval[1] - time_interval[0]
    scaled_time = (cur_time - time_interval[0]) / dt

    # define the polinomial
    acc = 0
    for i in range(2, len(coeffs)):
        acc += i * (i - 1) * coeffs[i] * (scaled_time ** (i - 2)) / dt**2

    return acc


def jerk_processor(coeffs, time_interval, cur_time):
    """
    This function returns the value of the third derivative of the polinomial at the given time.

    Parameters
    ----------
    coeffs : list
        The coefficients of the polinomial.
    time_interval : list
        The time interval of the polinomial.
    cur_time : float
        The time at which the polinomial should be evaluated.

    Returns
    -------
    float
        The value of the third derivative of the polinomial at the given time.
    """
    if cur_time[0] < time_interval[0] or cur_time[1] > time_interval[1]:
        raise ValueError(
            "The given time is not in the time interval of the polinomial."
        )

    dt = time_interval[1] - time_interval[0]
    scaled_time = (cur_time - time_interval[0]) / dt

    # define the polinomial
    jerk = 0
    for i in range(3, len(coeffs)):
        jerk += i * (i - 1) * (i - 2) * coeffs[i] * (scaled_time ** (i - 3)) / dt**3

    return jerk


def snap_processor(coeffs, time_interval, cur_time):
    """
    This function returns the value of the fourth derivative of the polinomial at the given time.

    Parameters
    ----------
    coeffs : list
        The coefficients of the polinomial.
    time_interval : list
        The time interval of the polinomial.
    cur_time : float
        The time at which the polinomial should be evaluated.

    Returns
    -------
    float
        The value of the fourth derivative of the polinomial at the given time.
    """
    if cur_time[0] < time_interval[0] or cur_time[1] > time_interval[1]:
        raise ValueError(
            "The given time is not in the time interval of the polinomial."
        )

    dt = time_interval[1] - time_interval[0]
    scaled_time = (cur_time - time_interval[0]) / dt

    # define the polinomial
    snap = 0
    for i in range(4, len(coeffs)):
        snap += (
            i
            * (i - 1)
            * (i - 2)
            * (i - 3)
            * coeffs[i]
            * (scaled_time ** (i - 4))
            / dt**4
        )

    return snap


def crackle_processor(coeffs, time_interval, cur_time):
    """
    This function returns the value of the fifth derivative of the polinomial at the given time.

    Parameters
    ----------
    coeffs : list
        The coefficients of the polinomial.
    time_interval : list
        The time interval of the polinomial.
    cur_time : float
        The time at which the polinomial should be evaluated.

    Returns
    -------
    float
        The value of the fifth derivative of the polinomial at the given time.
    """
    if cur_time[0] < time_interval[0] or cur_time[1] > time_interval[1]:
        raise ValueError(
            "The given time is not in the time interval of the polinomial."
        )

    dt = time_interval[1] - time_interval[0]
    scaled_time = (cur_time - time_interval[0]) / dt

    # define the polinomial
    crackle = 0
    for i in range(5, len(coeffs)):
        crackle += (
            i
            * (i - 1)
            * (i - 2)
            * (i - 3)
            * (i - 4)
            * coeffs[i]
            * (scaled_time ** (i - 5))
            / dt**5
        )

    return crackle

def pop_processor(coeffs, time_interval, cur_time):
    """
    This function returns the value of the sixth derivative of the polinomial at the given time.

    Parameters
    ----------
    coeffs : list
        The coefficients of the polinomial.
    time_interval : list
        The time interval of the polinomial.
    cur_time : float
        The time at which the polinomial should be evaluated.

    Returns
    -------
    float
        The value of the sixth derivative of the polinomial at the given time.
    """
    if cur_time[0] < time_interval[0] or cur_time[1] > time_interval[1]:
        raise ValueError(
            "The given time is not in the time interval of the polinomial."
        )

    dt = time_interval[1] - time_interval[0]
    scaled_time = (cur_time - time_interval[0]) / dt

    # define the polinomial
    pop = 0
    for i in range(6, len(coeffs)):
        pop += (
            i
            * (i - 1)
            * (i - 2)
            * (i - 3)
            * (i - 4)
            * (i - 5)
            * coeffs[i]
            * (scaled_time ** (i - 6))
            / dt**6
        )

    return pop