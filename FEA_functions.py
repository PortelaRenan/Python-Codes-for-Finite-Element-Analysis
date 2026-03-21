import numpy as np

def gaussPoints(n: int, a: float =-1, b: float = 1):
    """
    Compute Gauss-Legendre quadrature points and weights.
    
    Parameters:
        n : int   - number of Gauss points
        a : float - lower bound of integration (default -1)
        b : float - upper bound of integration (default  1)
    
    Returns:
        points  : array of quadrature points
        weights : array of quadrature weights
    """
    # Get points and weights on [-1, 1]
    points, weights = np.polynomial.legendre.leggauss(n)
    
    # Scale to [a, b]
    points  = 0.5 * (b - a) * points + 0.5 * (b + a)
    weights = 0.5 * (b - a) * weights
    
    return points, weights

def shapeFunctionL2(xi: float):
    """
    Compute shape function and derivatives for L2 elements

    Parameters:
        xi : natural coordinates (-1 ... +1)

    Returns:
        N: array of shape function values
        B: array  of natural derivatives values
    """
    # shape function
    N: list = [[(1 - xi)/2], [(1 + xi)/2]]

    # natural derivative
    B: list = [-1/2, 1/2]

    return N, B