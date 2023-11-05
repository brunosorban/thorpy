import warnings
import numpy as np

# Suppress RankWarning
warnings.simplefilter("ignore", np.RankWarning)


def estimate_coeffs(time, points, n):
    """Estimate coefficients of a polynomial of nth order. n is calculated based on the number of points.

    Args:
        points (list): list of points to be interpolated.
    """

    # Sample data points
    x = np.array(time)  # x-values
    y = np.array(points)  # y-values

    # Interpolate polynomial of 8th order
    coefficients = np.polyfit(x, y, n)

    return coefficients
