import numpy as np
import matplotlib.pyplot as plt

def gaussPoints(n: int, a: float =-1, b: float = 1) -> tuple[list[float], list[float]]:
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

def shapeFunctionL2(xi: float) -> tuple[np.ndarray, np.ndarray]:
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

def stiffness2Dtruss(elasticModulus: float, crossSection: float, GDOF: int, numberElement: int, elementNodes: list[tuple[int, int]], numberNodes: int, nodeCoordinates: list[tuple[float, float]]) -> np.ndarray:
    """
    Assemble the global stiffness matrix for a 2D truss structure.

    Parameters:
        elasticModulus  : Young's modulus of the material (E)
        crossSection    : Cross-sectional area of the truss elements (A)
        GDOF            : Total number of global degrees of freedom
        numberElement   : Number of truss elements
        elementNodes    : Connectivity table — each entry is a (node_i, node_j) pair
                          defining the two nodes of an element
        numberNodes     : Total number of nodes
        nodeCoordinates : Nodal coordinates — each entry is an (x, y) pair

    Returns:
        stiffness : np.ndarray, shape (GDOF, GDOF)
            Global stiffness matrix assembled from all element contributions
    """
    
    stiffness = np.zeros((GDOF, GDOF))

    for element in range(numberElement):
        indice = elementNodes[element]
        DOF = [2*indice[0], 2*indice[0] + 1, 2*indice[1], 2*indice[1] + 1]
        
        xCord = nodeCoordinates[indice[1]][0] - nodeCoordinates[indice[0]][0]
        yCord = nodeCoordinates[indice[1]][1] - nodeCoordinates[indice[0]][1]

        lengthElement = np.sqrt(xCord**2 + yCord**2)

        cosine = xCord/lengthElement
        sine   = yCord/lengthElement

        localStiffness = (elasticModulus * crossSection / lengthElement) * np.array([
                            [ cosine**2,      cosine*sine,  -cosine**2,     -cosine*sine],
                            [ cosine*sine,    sine**2,      -cosine*sine,   -sine**2    ],
                            [-cosine**2,     -cosine*sine,   cosine**2,      cosine*sine ],
                            [-cosine*sine,   -sine**2,       cosine*sine,    sine**2     ]
                        ])

        stiffness[np.ix_(DOF, DOF)] += localStiffness
    

    return stiffness

def stress2Dtruss(numberElements: int, elementNodes: list[tuple[int, int]], nodeCoordinates: list[tuple[float, float]], displacement: list[float], elasticModulus: float):
    """
    Compute the axial stress in each element of a 2D truss structure.

    Parameters:
        numberElements  : Number of truss elements
        elementNodes    : Connectivity table — each entry is a (node_i, node_j) pair
                          defining the two nodes of an element
        nodeCoordinates : Nodal coordinates — each entry is an (x, y) pair
        displacement    : Global displacement vector of length GDOF, ordered as
                          [u1, v1, u2, v2, ..., un, vn]
        elasticModulus  : Young's modulus of the material (E)

    Returns:
        sigma : np.ndarray, shape (numberElements,)
            Axial stress in each element. Positive values indicate tension,
            negative values indicate compression.

    Notes:
        Stress is computed as σ = (E / L) · [-c, -s, c, s] · u_e,
        where c and s are the direction cosines, L is the element length,
        and u_e is the local displacement vector extracted from the global
        displacement array.
    """
    
    sigma = np.zeros((numberElements))

    for element in range(numberElements):
        indice = elementNodes[element]
        DOF = [2*indice[0], 2*indice[0] + 1, 2*indice[1], 2*indice[1] + 1]    

        xCord = nodeCoordinates[indice[1]][0] - nodeCoordinates[indice[0]][0]
        yCord = nodeCoordinates[indice[1]][1] - nodeCoordinates[indice[0]][1]

        lengthElement = np.sqrt(xCord**2 + yCord**2)

        cosine = xCord/lengthElement
        sine   = yCord/lengthElement

        sigma[element] = elasticModulus/lengthElement*np.dot(np.array([[-cosine, -sine, cosine, sine]]), displacement[DOF])[0]
    
    return sigma

def mesh(numberElements: int, elementNodes: list[tuple[int, int]], nodeCoordinates: list[tuple[float, float]], displacement: list[float],) -> None:
    """
    Plot the original and deformed configurations of a 2D truss structure.

    Parameters:
        numberElements  : Number of truss elements
        elementNodes    : Connectivity table — each entry is a (node_i, node_j) pair
                          defining the two nodes of an element
        nodeCoordinates : Nodal coordinates — each entry is an (x, y) pair
        displacement    : Global displacement vector of length GDOF, ordered as
                          [u1, v1, u2, v2, ..., un, vn]

    Returns:
        None — displays a matplotlib figure.

    Notes:
        - Original nodes and members are drawn in black and blue, respectively.
        - Deformed nodes and members are drawn in red (solid and dashed lines).
        - Displacements are scaled by a factor of 5e2 for visualization purposes.
    """

    plt.figure()

    nodes = np.array(nodeCoordinates)
    plt.scatter(nodes[:, 0], nodes[:, 1], color = 'black')

    displacements = displacement.reshape((-1, 2))*1e2
    newCoord = nodeCoordinates + displacements
    plt.scatter(newCoord[:, 0], newCoord[:, 1], color = 'red')

    undeformed_plotted = False  
    deformed_plotted = False    

    for element in range(numberElements):
        nodes = elementNodes[element]
        
        x1 = nodeCoordinates[nodes[0]][0]
        y1 = nodeCoordinates[nodes[0]][1]

        x2 = nodeCoordinates[nodes[1]][0]
        y2 = nodeCoordinates[nodes[1]][1]

        plt.plot([x1, x2], [y1, y2], c = 'blue',
                 label='Undeformed' if not undeformed_plotted else '_nolegend_')
        undeformed_plotted = True

        x1 = newCoord[nodes[0]][0]
        y1 = newCoord[nodes[0]][1]

        x2 = newCoord[nodes[1]][0]
        y2 = newCoord[nodes[1]][1]

        plt.plot([x1, x2], [y1, y2], c = 'red', linestyle='dashed',
                 label='Deformed' if not deformed_plotted else '_nolegend_')
        deformed_plotted = True

    plt.legend()
    plt.show()