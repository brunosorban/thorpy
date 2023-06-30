import numpy as np


def estimate_coeffs(time, points, n=11):
    """Estimate coefficients of a polynomial of nth order. n is calculated based on the number of points.

    Args:
        points (list): list of points to be interpolated.
    """

    # Sample data points
    x = np.array(time)  # x-values
    y = np.array(points)  # y-values

    # Interpolate polynomial of 8th order
    coefficients = np.polyfit(x, y, n)
    print(coefficients)

    return coefficients
