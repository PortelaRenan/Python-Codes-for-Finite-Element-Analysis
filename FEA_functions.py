import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

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

def stress2Dtruss(numberElements: int, elementNodes: list[tuple[int, int]], nodeCoordinates: list[tuple[float, float]], displacement: list[float], elasticModulus: float) -> np.ndarray:
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

def BernoulliBeam(GDOF: int, elementNodes: list[tuple[int, int]], nodeCoordinates: list[tuple[float, float]], E: float, I: float) -> np.ndarray:
    """
    Assemble the global stiffness matrix for an Euler-Bernoulli beam structure.

    Parameters:
        GDOF            : Total number of global degrees of freedom
        elementNodes    : Connectivity table — each entry is a (node_i, node_j) pair
                          defining the two nodes of an element
        nodeCoordinates : Nodal coordinates — each entry is an x coordinate value
        E               : Young's modulus of the material
        I               : Second moment of area of the beam cross-section

    Returns:
        stiffness : np.ndarray, shape (GDOF, GDOF)
            Global stiffness matrix assembled from all element contributions
    """


    stiffness = np.zeros((GDOF, GDOF))

    for element in range(np.size(elementNodes,0)):
        indices = elementNodes[element]
        DOF = [2*indices[0], 2*indices[0] + 1, 2*indices[1], 2*indices[1] + 1]    

        length = nodeCoordinates[indices[1]] - nodeCoordinates[indices[0]]
        localStiffness = E*I/length**3*np.array([
            [12, 6*length, -12, 6*length],
            [6*length, 4*length**2, -6*length, 2*length**2],
            [-12, -6*length, 12, -6*length],
            [6*length, 2*length**2, -6*length, 4*length**2]
        ])

        stiffness[np.ix_(DOF, DOF)] += localStiffness

    return stiffness

def BernoulliBeamBC(bc: str = 'Simply supported', numberElements: int = None) -> np.ndarray:
    """
    Return the fixed degrees of freedom for a given Euler-Bernoulli beam boundary condition.

    Parameters:
        bc              : str — boundary condition type. Must be one of:
                          'Simply supported', 'Clamped-clamped', or 'Clamped'
                          (default: 'Simply supported')
        numberElements  : int — number of beam elements (must be a positive integer)

    Returns:
        fixedDOF : list of int
            Indices of the constrained degrees of freedom in the global DOF vector

    Raises:
        ValueError : if bc is not a valid boundary condition string
        ValueError : if numberElements is None or non-positive
    """

    valid_bc = ['Simply supported', 'Clamped-clamped', 'Clamped']
    
    if bc not in valid_bc:
        raise ValueError(
            f"Invalid boundary condition '{bc}'. Valid options are: {valid_bc}."
        )
    
    if numberElements is None or numberElements <= 0:
        raise ValueError("numberElements must be a positive integer.")
    
    if bc == 'Simply supported':
        fixedDOF = [0, 2 * numberElements]
    elif bc == 'Clamped-clamped':
        fixedDOF = [0, 1, 2 * numberElements, 2 * numberElements + 1]
    else:  # Clamped at x = 0
        fixedDOF = [0, 1]  

    return fixedDOF

def distributedLoad(GDOF: int, elementNodes: list[tuple[int, int]], nodeCoordinates: list[tuple[float, float]], P: float) -> np.ndarray:
    """
    Assemble the global load vector for a uniformly distributed load on a beam.

    Parameters:
        GDOF            : Total number of global degrees of freedom
        elementNodes    : Connectivity table — each entry is a (node_i, node_j) pair
                          defining the two nodes of an element
        nodeCoordinates : Nodal coordinates — each entry is an x coordinate value
        P               : Magnitude of the uniformly distributed load

    Returns:
        load : np.ndarray, shape (GDOF,)
            Global equivalent nodal load vector assembled from all element contributions
    """

    load = np.zeros(GDOF)
    for element in range(np.size(elementNodes,0)):
        indices = elementNodes[element]
        DOF = [2*indices[0], 2*indices[0] + 1, 2*indices[1], 2*indices[1] + 1]    

        length = nodeCoordinates[indices[1]] - nodeCoordinates[indices[0]]
        
        localLoad = P*length*np.array([1/2, length/12, 1/2, -length/12])

        load[DOF] += localLoad

    return load

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

def isotropicMaterial(E, thickness, poisson):
    D11 = E*thickness**3/12/(1-poisson**2)
    D22 = D11; 
    D12 = poisson*D11
    D66 = (1-poisson)*D11/2

    return np.array([[D11, D12, 0],
                     [D12, D22, 0],
                     [0, 0, D66]])

def orthotropicMaterial(E1, E2, G12, poisson12, thickness):
    poisson21 = poisson12*E2/E1
    I = thickness**3/12
    D11 = E1*I/(1-poisson12*poisson21)
    D22 = E2*I/(1-poisson12*poisson21)
    D12 = poisson12*D22
    D66 = G12*I

    return np.array([[D11, D12, 0],
                     [D12, D22, 0],
                     [0, 0, D66]])

def rectangularMesh(Lx,Ly,numberElementsX,numberElementsY, elementType):
    if elementType == 'Q4':
        x = np.linspace(0, Lx, numberElementsX + 1)
        y = np.linspace(0, Ly, numberElementsY + 1)
        N = np.arange(0, (numberElementsX + 1)*(numberElementsY + 1), dtype= object)
        N = N.reshape(numberElementsX + 1,numberElementsY + 1)

        X, Y = np.meshgrid(x, y)

        return np.stack((X.flatten(), Y.flatten()), axis=1), [[N[i, j], N[i, j+1], N[i+1, j+1], N[i+1, j]] for i in range(numberElementsX) for j in range(numberElementsY)]   


def drawingMesh(nodeCoordinates, elementNodes, elementType):

    if elementType == 'Q4':
        # Build array of polygon vertices for all elements at once
        verts = [nodeCoordinates[element, :] for element in elementNodes]

        # Plot all elements in one call
        fig, ax = plt.subplots()
        mesh = PolyCollection(verts, edgecolors='black', facecolors='none', linewidths=0.5)
        ax.add_collection(mesh)

        ax.autoscale()
        ax.set_aspect('equal')
        plt.show()
        
def plateKirchhoffStiffness(GDOF, elementNodes, elementType, nodeCoordinates, dofPerNode):
    stiffness = np.zeros((GDOF, GDOF))

    points, weights = gaussPoints(elementType)

    XP, YP = np.meshgrid(points, points)
    XW, YW = np.meshgrid(weights, weights)
    
    gsPoints = np.stack((XP.flatten(), YP.flatten()), axis=1)
    gsWeights = np.stack((XW.flatten(), YW.flatten()), axis=1)

    # cycle for element
    for nodes in elementNodes:
        if dofPerNode == 3:
            DOF = [nodes, 
                   nodes + len(nodeCoordinates), 
                   nodes + 2*len(nodeCoordinates)]
        elif dofPerNode == 4:
            DOF = [nodes, 
                   nodes + len(nodeCoordinates), 
                   nodes + 2*len(nodeCoordinates),
                   nodes + 3*len(nodeCoordinates)]
      
        # cycle for Gauss point
        for p, w in zip(gsPoints, gsWeights):
            # part related to the mapping
            # shape functions and derivatives
            _,natDerQ4 = shapeFunctionQ4(p[0],p[1])

            if dofPerNode == 3:
                _, naturalDerivatives = shapeFunctionNotConforming(p[0], p[1])
            elif dofPerNode == 4:
                _, naturalDerivatives = shapeFunctionConforming(p[0], p[1])
            
            Jacob, _ = Jacobian(nodes, natDerQ4)
            XYDer = nodes.T @ natDerQ4

            detJ_xi = XYDer[0,0]*XYDer[1,4] - XYDer[1,0]*XYDer[0,4]
            + XYDer[1,1]*XYDer[0,2] - XYDer[0,1]*XYDer[1,2]
            detJ_eta = -XYDer[0,1]*XYDer[1,4] + XYDer[1,1]*XYDer[0,4]
            - XYDer[1,0]*XYDer[0,3] + XYDer[0,0]*XYDer[1,3]

            xi_xx = np.linalg.det(Jacob)**(-2)*(XYDer[1,1]*XYDer[1,4]
            - XYDer[1,1]**2/np.linalg.det(Jacob)*detJ_xi - XYDer[1,0]*XYDer[1,3] 
            + XYDer[1,0]*XYDer[1,1]/np.linalg.det(Jacob)*detJ_eta)

            xi_yy = np.linalg.det(Jacob)**(-2)*(XYDer[0,1]*XYDer[0,4] - XYDer[0,1]**2/np.linalg.det(Jacob)*detJ_xi
            - XYDer[0,0]*XYDer[0,3] + XYDer[0,0]*XYDer[0,1]/np.linalg.det(Jacob)*detJ_eta)

            eta_xx = np.linalg.det(Jacob)**(-2)*(-XYDer[1,1]*XYDer[1,2] + XYDer[1,1]*XYDer[1,0]/np.linalg.det(Jacob)*detJ_xi 
            + XYDer[1,0]*XYDer[1,4] - XYDer[1,0]**2/np.linalg.det(Jacob)*detJ_eta)

            eta_yy = np.linalg.det(Jacob)**(-2)*(-XYDer[0,1]*XYDer[0,2] + XYDer[0,1]*XYDer[0,0]/np.linalg.det(Jacob)*detJ_xi 
            + XYDer[0,0]*XYDer[0,4] - XYDer[0,0]**2/np.linalg.det(Jacob)*detJ_eta)

            xi_xy = np.linalg.det(Jacob)**(-2)*(-XYDer[1,1]*XYDer[0,4] + XYDer[1,1]*XYDer[0,1]/np.linalg.det(Jacob)*detJ_xi 
            + XYDer[1,0]*XYDer[0,3] - XYDer[1,0]*XYDer[0,1]/np.linalg.det(Jacob)*detJ_eta)

            eta_xy = np.linalg.det(Jacob)**(-2)*(-XYDer[1,0]*XYDer[0,4] - XYDer[1,1]*XYDer[0,0]/np.linalg.det(Jacob)*detJ_xi 
            + XYDer[1,1]*XYDer[0,2] + XYDer[1,0]*XYDer[0,0]/np.linalg.det(Jacob)*detJ_eta)

            # [B] matrix bending
            xi_x = np.linalg.inv(Jacob)[0,0]
            xi_y = np.linalg.inv(Jacob)[1,0]
            eta_x = np.linalg.inv(Jacob)[0,1]
            eta_y = np.linalg.inv(Jacob)[1,1]
            
            B_b = np.zeros((3, GDOF))

            B_b[0, 0:GDOF] = (
                xi_x**2 * naturalDerivatives[:, 2].T +
                eta_x**2 * naturalDerivatives[:, 3].T +
                2 * xi_x * eta_x * naturalDerivatives[:, 4].T +
                xi_xx * naturalDerivatives[:, 0].T +
                eta_xx * naturalDerivatives[:, 1].T
            )

            B_b[1, 0:GDOF] = (
                xi_y**2 * naturalDerivatives[:, 2].T +
                eta_y**2 * naturalDerivatives[:, 3].T +
                2 * xi_y * eta_y * naturalDerivatives[:, 4].T +
                xi_yy * naturalDerivatives[:, 0].T +
                eta_yy * naturalDerivatives[:, 1].T
            )

            B_b[2, 0:GDOF] = 2 * (
                xi_x * xi_y * naturalDerivatives[:, 2].T +
                eta_x * eta_y * naturalDerivatives[:, 3].T +
                (xi_x * eta_y + xi_y * eta_x) * naturalDerivatives[:, 4].T +
                xi_xy * naturalDerivatives[:, 0].T +
                eta_xy * naturalDerivatives[:, 1].T
            )



            # stiffness matrix bending
            stiffness[np.ix_(DOF, DOF)] +=B_b.T*C_bending*B_b*gaussWeights(q)*det(Jacob)

    raise NotImplemented

def plateBC(typeBC, nodeCoordinates, GDOF):
    xx = nodeCoordinates[:, 0]
    yy = nodeCoordinates[:, 1]
    
    if typeBC == 'ssss':      
        fixedNodeW = np.where(
            (xx == xx.max()) |
            (xx == xx.min()) |
            (yy == yy.min()) |
            (yy == yy.max())
        )[0]

        fixedNodeTX = np.where(
            (yy == yy.min()) |
            (yy == yy.max())
        )[0]

        fixedNodeTY = np.where(
            (xx == xx.min()) |
            (xx == xx.max())
        )[0]
    elif typeBC == 'cccc':
        fixedNodeW = np.where(
            (xx == xx.max()) |
            (xx == xx.min()) |
            (yy == yy.min()) |
            (yy == yy.max())
        )[0]

        fixedNodeTX = fixedNodeW

        fixedNodeTY = fixedNodeW
    elif typeBC == 'scsc':
        fixedNodeW = np.where(
            (xx == xx.max()) |
            (xx == xx.min()) |
            (yy == yy.min()) |
            (yy == yy.max())
        )[0]

        fixedNodeTX = np.where(
            (xx == xx.min()) |
            (xx == xx.max())
        )[0]

        fixedNodeTY = np.array([])
    elif typeBC == 'cccf':    
        fixedNodeW = np.where(
            (xx == xx.min()) |
            (yy == yy.min()) |
            (yy == yy.max())
        )[0]
        fixedNodeTX = fixedNodeW

        fixedNodeTY = fixedNodeW

    fixedDOF = np.concatenate([
        fixedNodeW, 
        fixedNodeTX + len(nodeCoordinates),
        fixedNodeTY + 2*len(nodeCoordinates)
    ])
    
    mask = np.arange(GDOF)
    return np.bool([0 if x in fixedDOF else 1 for x in mask])

def shapeFunctionNotConforming(xi, eta):
    shape = np.array([
        -((eta - 1)*(xi - 1)*(eta**2 + eta + xi**2 + xi - 2))/8,
        ((eta - 1)*(xi + 1)*(eta**2 + eta + xi**2 - xi - 2))/8,
        ((eta + 1)*(xi + 1)*(-eta**2 + eta - xi**2 + xi + 2))/8,
        ((eta + 1)*(xi - 1)*(eta**2 - eta + xi**2 + xi - 2))/8,
        -((eta - 1)*(xi - 1)**2*(xi + 1))/8,
        -((eta - 1)*(xi - 1)*(xi + 1)**2)/8,
        ((eta + 1)*(xi - 1)*(xi + 1)**2)/8,
        ((eta + 1)*(xi - 1)**2*(xi + 1))/8,
        -((eta - 1)**2*(eta + 1)*(xi - 1))/8,
        ((eta - 1)**2*(eta + 1)*(xi + 1))/8,
        ((eta - 1)*(eta + 1)**2*(xi + 1))/8,
        -((eta - 1)*(eta + 1)**2*(xi - 1))/8
    ])

    naturalDerivatives = np.zeros((12, 5))

    naturalDerivatives[:, 0] = [
        -((eta - 1)*(eta**2 + eta + 3*xi**2 - 3))/8,
        ((eta - 1)*(eta**2 + eta + 3*xi**2 - 3))/8,
        ((eta + 1)*(-eta**2 + eta - 3*xi**2 + 3))/8,
        -((eta + 1)*(-eta**2 + eta - 3*xi**2 + 3))/8,
        ((eta - 1)*(-3*xi**2 + 2*xi + 1))/8,
        -((eta - 1)*(3*xi**2 + 2*xi - 1))/8,
        ((eta + 1)*(3*xi**2 + 2*xi - 1))/8,
        -((eta + 1)*(-3*xi**2 + 2*xi + 1))/8,
        -((eta - 1)**2*(eta + 1))/8,
        ((eta - 1)**2*(eta + 1))/8,
        ((eta - 1)*(eta + 1)**2)/8,
        -((eta - 1)*(eta + 1)**2)/8
    ]

    naturalDerivatives[:, 1] = [
        -((xi - 1)*(3*eta**2 + xi**2 + xi - 3))/8,
        -((xi + 1)*(-3*eta**2 - xi**2 + xi + 3))/8,
        ((xi + 1)*(-3*eta**2 - xi**2 + xi + 3))/8,
        ((xi - 1)*(3*eta**2 + xi**2 + xi - 3))/8,
        -((xi - 1)**2*(xi + 1))/8,
        -((xi - 1)*(xi + 1)**2)/8,
        ((xi - 1)*(xi + 1)**2)/8,
        ((xi - 1)**2*(xi + 1))/8,
        ((xi - 1)*(-3*eta**2 + 2*eta + 1))/8,
        -((xi + 1)*(-3*eta**2 + 2*eta + 1))/8,
        ((xi + 1)*(3*eta**2 + 2*eta - 1))/8,
        -((xi - 1)*(3*eta**2 + 2*eta - 1))/8
    ]

    naturalDerivatives[:, 2] = [
        -(3*xi*(eta - 1))/4,
        (3*xi*(eta - 1))/4,
        -(3*xi*(eta + 1))/4,
        (3*xi*(eta + 1))/4,
        -((3*xi - 1)*(eta - 1))/4,
        -((3*xi + 1)*(eta - 1))/4,
        ((3*xi + 1)*(eta + 1))/4,
        ((3*xi - 1)*(eta + 1))/4,
        0, 0, 0, 0
    ]

    naturalDerivatives[:, 3] = [
        -(3*eta*(xi - 1))/4,
        (3*eta*(xi + 1))/4,
        -(3*eta*(xi + 1))/4,
        (3*eta*(xi - 1))/4,
        0, 0, 0, 0,
        -((3*eta - 1)*(xi - 1))/4,
        ((3*eta - 1)*(xi + 1))/4,
        ((3*eta + 1)*(xi + 1))/4,
        -((3*eta + 1)*(xi - 1))/4
    ]

    naturalDerivatives[:, 4] = [
        1/2 - (3*xi**2)/8 - (3*eta**2)/8,
        (3*eta**2)/8 + (3*xi**2)/8 - 1/2,
        1/2 - (3*xi**2)/8 - (3*eta**2)/8,
        (3*eta**2)/8 + (3*xi**2)/8 - 1/2,
        xi/4 - (3*xi**2)/8 + 1/8,
        1/8 - (3*xi**2)/8 - xi/4,
        (3*xi**2)/8 + xi/4 - 1/8,
        (3*xi**2)/8 - xi/4 - 1/8,
        eta/4 - (3*eta**2)/8 + 1/8,
        (3*eta**2)/8 - eta/4 - 1/8,
        (3*eta**2)/8 + eta/4 - 1/8,
        1/8 - (3*eta**2)/8 - eta/4
    ]

    return shape, naturalDerivatives

def shapeFunctionConforming(xi, eta):
    shape = np.array([
        ((eta - 1)**2*(eta + 2)*(xi - 1)**2*(xi + 2))/16,
        -((eta - 1)**2*(eta + 2)*(xi + 1)**2*(xi - 2))/16,
        ((eta + 1)**2*(eta - 2)*(xi + 1)**2*(xi - 2))/16,
        -((eta + 1)**2*(eta - 2)*(xi - 1)**2*(xi + 2))/16,
        ((eta - 1)**2*(eta + 2)*(xi - 1)**2*(xi + 1))/16,
        ((eta - 1)**2*(eta + 2)*(xi - 1)*(xi + 1)**2)/16,
        -((eta + 1)**2*(eta - 2)*(xi - 1)*(xi + 1)**2)/16,
        -((eta + 1)**2*(eta - 2)*(xi - 1)**2*(xi + 1))/16,
        ((eta - 1)**2*(eta + 1)*(xi - 1)**2*(xi + 2))/16,
        -((eta - 1)**2*(eta + 1)*(xi + 1)**2*(xi - 2))/16,
        -((eta - 1)*(eta + 1)**2*(xi + 1)**2*(xi - 2))/16,
        ((eta - 1)*(eta + 1)**2*(xi - 1)**2*(xi + 2))/16,
        ((eta - 1)**2*(eta + 1)*(xi - 1)**2*(xi + 1))/16,
        ((eta - 1)**2*(eta + 1)*(xi - 1)*(xi + 1)**2)/16,
        ((eta - 1)*(eta + 1)**2*(xi - 1)*(xi + 1)**2)/16,
        ((eta - 1)*(eta + 1)**2*(xi - 1)**2*(xi + 1))/16
    ])

    naturalDerivatives = np.zeros((16, 5))

    naturalDerivatives[:, 0] = [
        (3*(xi**2 - 1)*(eta - 1)**2*(eta + 2))/16,
        -(3*(xi**2 - 1)*(eta - 1)**2*(eta + 2))/16,
        (3*(xi**2 - 1)*(eta + 1)**2*(eta - 2))/16,
        -(3*(xi**2 - 1)*(eta + 1)**2*(eta - 2))/16,
        -((eta - 1)**2*(eta + 2)*(-3*xi**2 + 2*xi + 1))/16,
        ((eta - 1)**2*(eta + 2)*(3*xi**2 + 2*xi - 1))/16,
        -((eta + 1)**2*(eta - 2)*(3*xi**2 + 2*xi - 1))/16,
        ((eta + 1)**2*(eta - 2)*(-3*xi**2 + 2*xi + 1))/16,
        (3*(xi**2 - 1)*(eta - 1)**2*(eta + 1))/16,
        -(3*(xi**2 - 1)*(eta - 1)**2*(eta + 1))/16,
        -(3*(xi**2 - 1)*(eta - 1)*(eta + 1)**2)/16,
        (3*(xi**2 - 1)*(eta - 1)*(eta + 1)**2)/16,
        -((eta - 1)**2*(eta + 1)*(-3*xi**2 + 2*xi + 1))/16,
        ((eta - 1)**2*(eta + 1)*(3*xi**2 + 2*xi - 1))/16,
        ((eta - 1)*(eta + 1)**2*(3*xi**2 + 2*xi - 1))/16,
        -((eta - 1)*(eta + 1)**2*(-3*xi**2 + 2*xi + 1))/16
    ]

    naturalDerivatives[:, 1] = [
        (3*(eta**2 - 1)*(xi - 1)**2*(xi + 2))/16,
        -(3*(eta**2 - 1)*(xi + 1)**2*(xi - 2))/16,
        (3*(eta**2 - 1)*(xi + 1)**2*(xi - 2))/16,
        -(3*(eta**2 - 1)*(xi - 1)**2*(xi + 2))/16,
        (3*(eta**2 - 1)*(xi - 1)**2*(xi + 1))/16,
        (3*(eta**2 - 1)*(xi - 1)*(xi + 1)**2)/16,
        -(3*(eta**2 - 1)*(xi - 1)*(xi + 1)**2)/16,
        -(3*(eta**2 - 1)*(xi - 1)**2*(xi + 1))/16,
        -((xi - 1)**2*(xi + 2)*(-3*eta**2 + 2*eta + 1))/16,
        ((xi + 1)**2*(xi - 2)*(-3*eta**2 + 2*eta + 1))/16,
        -((xi + 1)**2*(xi - 2)*(3*eta**2 + 2*eta - 1))/16,
        ((xi - 1)**2*(xi + 2)*(3*eta**2 + 2*eta - 1))/16,
        -((xi - 1)**2*(xi + 1)*(-3*eta**2 + 2*eta + 1))/16,
        -((xi - 1)*(xi + 1)**2*(-3*eta**2 + 2*eta + 1))/16,
        ((xi - 1)*(xi + 1)**2*(3*eta**2 + 2*eta - 1))/16,
        ((xi - 1)**2*(xi + 1)*(3*eta**2 + 2*eta - 1))/16
    ]

    naturalDerivatives[:, 2] = [
        (3*xi*(eta - 1)**2*(eta + 2))/8,
        -(3*xi*(eta - 1)**2*(eta + 2))/8,
        (3*xi*(eta + 1)**2*(eta - 2))/8,
        -(3*xi*(eta + 1)**2*(eta - 2))/8,
        ((3*xi - 1)*(eta - 1)**2*(eta + 2))/8,
        ((3*xi + 1)*(eta - 1)**2*(eta + 2))/8,
        -((3*xi + 1)*(eta + 1)**2*(eta - 2))/8,
        -((3*xi - 1)*(eta + 1)**2*(eta - 2))/8,
        (3*xi*(eta - 1)**2*(eta + 1))/8,
        -(3*xi*(eta - 1)**2*(eta + 1))/8,
        -(3*xi*(eta - 1)*(eta + 1)**2)/8,
        (3*xi*(eta - 1)*(eta + 1)**2)/8,
        ((3*xi - 1)*(eta - 1)**2*(eta + 1))/8,
        ((3*xi + 1)*(eta - 1)**2*(eta + 1))/8,
        ((3*xi + 1)*(eta - 1)*(eta + 1)**2)/8,
        ((3*xi - 1)*(eta - 1)*(eta + 1)**2)/8
    ]

    naturalDerivatives[:, 3] = [
        (3*eta*(xi - 1)**2*(xi + 2))/8,
        -(3*eta*(xi + 1)**2*(xi - 2))/8,
        (3*eta*(xi + 1)**2*(xi - 2))/8,
        -(3*eta*(xi - 1)**2*(xi + 2))/8,
        (3*eta*(xi - 1)**2*(xi + 1))/8,
        (3*eta*(xi - 1)*(xi + 1)**2)/8,
        -(3*eta*(xi - 1)*(xi + 1)**2)/8,
        -(3*eta*(xi - 1)**2*(xi + 1))/8,
        ((3*eta - 1)*(xi - 1)**2*(xi + 2))/8,
        -((3*eta - 1)*(xi + 1)**2*(xi - 2))/8,
        -((3*eta + 1)*(xi + 1)**2*(xi - 2))/8,
        ((3*eta + 1)*(xi - 1)**2*(xi + 2))/8,
        ((3*eta - 1)*(xi - 1)**2*(xi + 1))/8,
        ((3*eta - 1)*(xi - 1)*(xi + 1)**2)/8,
        ((3*eta + 1)*(xi - 1)*(xi + 1)**2)/8,
        ((3*eta + 1)*(xi - 1)**2*(xi + 1))/8
    ]

    naturalDerivatives[:, 4] = [
        (9*(eta**2 - 1)*(xi**2 - 1))/16,
        -(9*(eta**2 - 1)*(xi**2 - 1))/16,
        (9*(eta**2 - 1)*(xi**2 - 1))/16,
        -(9*(eta**2 - 1)*(xi**2 - 1))/16,
        -(3*(eta**2 - 1)*(-3*xi**2 + 2*xi + 1))/16,
        (3*(eta**2 - 1)*(3*xi**2 + 2*xi - 1))/16,
        -(3*(eta**2 - 1)*(3*xi**2 + 2*xi - 1))/16,
        (3*(eta**2 - 1)*(-3*xi**2 + 2*xi + 1))/16,
        -(3*(xi**2 - 1)*(-3*eta**2 + 2*eta + 1))/16,
        (3*(xi**2 - 1)*(-3*eta**2 + 2*eta + 1))/16,
        -(3*(xi**2 - 1)*(3*eta**2 + 2*eta - 1))/16,
        (3*(xi**2 - 1)*(3*eta**2 + 2*eta - 1))/16,
        ((-3*eta**2 + 2*eta + 1)*(-3*xi**2 + 2*xi + 1))/16,
        -((-3*eta**2 + 2*eta + 1)*(3*xi**2 + 2*xi - 1))/16,
        ((3*eta**2 + 2*eta - 1)*(3*xi**2 + 2*xi - 1))/16,
        -((3*eta**2 + 2*eta - 1)*(-3*xi**2 + 2*xi + 1))/16
    ]

    return shape, naturalDerivatives

def Jacobian(nodeCoordinates, naturalDerivatives):  
    return np.matmul(nodeCoordinates.T, naturalDerivatives[:, 0:2]), np.matmul(nodeCoordinates.T, naturalDerivatives)

def shapeFunctionQ4(xi, eta):
    shapeFunction = 1/4*np.array([[(1 - xi)*(1 - eta)],
                         [(1 + xi)*(1 - eta)],
                         [(1 + xi)*(1 + eta)],
                         [(1 - xi)*(1 + eta)]
    ]) 
    
    # natural derivatives order:
    # [d/dx, d/dy, d^2/dx^2, d^2/dy^2, d^2/dxdy]
    naturalDerivatives = (1/4) * np.array([
        [-(1 - eta), -(1 - xi)],
        [ (1 - eta), -(1 + xi)],
        [ (1 + eta),  (1 + xi)],
        [-(1 + eta),  (1 - xi)]
    ])

    # Add 5th column (d^2/dxdy)
    col5 = np.array([1/4, -1/4, 1/4, -1/4]).reshape(4, 1)

    # Combine into final (4 x 5) matrix
    naturalDerivatives = np.hstack((naturalDerivatives, col5))

    return shapeFunction, naturalDerivatives

def forceVector(GDOF, elementNodes, nodeCoordinates, P, elementType, dofPerNode):
    force = np.zeros(GDOF)
    
    if elementType == 'Q4':
        n = 2
    elif elementType == 'Q9':
        n = 3

    gp_coords, gp_weights = gaussPoints(n)
    XP, YP = np.meshgrid(gp_coords, gp_coords)
    XW, YW = np.meshgrid(gp_weights, gp_weights)
    
    gaussPoints = np.stack((XP.flatten(), YP.flatten()), axis=1)
    gaussWeights = np.stack((XW.flatten(), YW.flatten()), axis=1)

    # cycle for element
    for element in range(0, len(elementNodes)): 
        nodes = nodeCoordinates[element, :]
        
        if dofPerNode == 3:
            DOF = [nodes, 
                   nodes + len(nodeCoordinates), 
                   nodes + 2*len(nodeCoordinates)]
            
        elif dofPerNode == 4:
            DOF = [nodes, 
                   nodes + len(nodeCoordinates), 
                   nodes + 2*len(nodeCoordinates),
                   nodes + 3*len(nodeCoordinates)]

        # cycle for Gauss point
        for p, w in zip(gaussPoints, gaussWeights):
            # shape functions and derivatives
            _,natDerQ4 = shapeFunctionQ4(p[0],p[1])

            if dofPerNode == 3:
                shapeFunction, _ = shapeFunctionNotConforming(p[0], p[1])
            elif dofPerNode == 4:
                shapeFunction, _ = shapeFunctionConforming(p[0], p[1])

            Jacob, _ = Jacobian(nodes, natDerQ4)

            # force vector assembly
            force[DOF] += shapeFunction*P*np.linalg.det(Jacob)*w

    return force

def massMatrixTimoshenko(rho, thickness, nodeCoordinates, elementNodes, GDOF, n = 2):

    numberNodes = np.size(nodeCoordinates,0)

    # computation of stiffness matrix  for Timoshenko beam element
    mass = np.zeros((GDOF, GDOF))
    I = thickness**3/12
    gp_coords, gp_weights = gaussPoints(n)

    # bending contribution for stiffness matrix
    for i in range(len(elementNodes) - 1):
        indice = elementNodes[i]
        massDOF = [indice[0] + numberNodes, indice[1] + numberNodes]
        lengthElement = nodeCoordinates[indice[1]] - nodeCoordinates[indice[0]]
        detJacobian = lengthElement/2
        
        for j in range(n):
            point = gp_coords[j]
            N, _ = shapeFunctionL2(point)

            mass[np.ix_(massDOF, massDOF)] += np.array(N)@ np.array(N).T*I*rho*detJacobian*gp_weights[j]
            mass[np.ix_(indice, indice)] += np.array(N)@ np.array(N).T*thickness*rho*detJacobian*gp_weights[j]

    return mass


def forceTimoshenko(GDOF, nodeCoordinates, elementNodes, P, n = 2):
    
    force = np.zeros(GDOF)

    numberNodes = np.size(nodeCoordinates,0)

    gp_coords, gp_weights = gaussPoints(n)

    # bending contribution for stiffness matrix
    for i in range(len(elementNodes) - 1):
        indice = elementNodes[i]
        
        DOF = [indice[0], indice[1]]
        
        lengthElement = nodeCoordinates[DOF[1]] - nodeCoordinates[DOF[0]]
        detJacobian = lengthElement/2

        for j in range(2):
            point = gp_coords[j]
            N, _ = shapeFunctionL2(point)

            # Force vector        
            force[DOF] += (np.array(N)*P*detJacobian*gp_weights[j]).flatten()
            
    return force

def stiffnessTimoshenko(C, GDOF, elementNodes, nodeCoordinates, n = 2):
    
    # computation of stiffness matrix  for Timoshenko beam element
    stiffness = np.zeros((GDOF, GDOF))

    numberNodes = np.size(nodeCoordinates,0)

    gp_coords, gp_weights = gaussPoints(n)

    # bending contribution for stiffness matrix
    for i in range(len(elementNodes) - 1):
        indice = elementNodes[i]
        
        DOF = [indice[0], indice[1], indice[0] + numberNodes, indice[1] + numberNodes]
        
        lengthElement = nodeCoordinates[DOF[1]] - nodeCoordinates[DOF[0]]
        detJacobian = lengthElement/2

        for j in range(n):
            point = gp_coords[j]
            N, naturalDerivative = shapeFunctionL2(point)
            Xderivatives = naturalDerivative/detJacobian

            # B matrix
            B = np.zeros((2,4))
            B[np.ix_([0], [2, 3])] = Xderivatives

            # Stiffness matrix
            stiffness[np.ix_(DOF, DOF)] += B.T @ B*gp_weights[j]*detJacobian*C[0,0]


    # shear contribution for stiffness matrix
    gp_coords, gp_weights = gaussPoints(1)

    for i in range(len(elementNodes) - 1):
        indice = elementNodes[i]
        DOF = [indice[0], indice[1], indice[0] + numberNodes, indice[1] + numberNodes]
        lengthElement = nodeCoordinates[DOF[1]] - nodeCoordinates[DOF[0]]
        detJacobian = lengthElement/2
        for j in range(1):
            point = gp_coords[j]
            N, naturalDerivative = shapeFunctionL2(point)
            Xderivative = naturalDerivative/detJacobian
            
            # B matrix
            B = np.zeros((2,4))
            B[np.ix_([0], [0, 1])] = Xderivatives
            B[np.ix_([0], [2, 3])] = np.array(N).T
            
            # Stiffness matrix
            stiffness[np.ix_(DOF, DOF)] += B.T @ B*gp_weights[j]*detJacobian*C[1,1]

    return stiffness

def fmt_array(arr: np.ndarray) -> str:
        return np.array2string(arr, precision=2, suppress_small=True, separator=", ")

def resultTable(displacement, reaction, nameProblem, clear_screen=True, width=120, **others):
    if clear_screen:
        os.system("cls" if os.name == "nt" else "clear")

    separator = "=" * width

    print(separator)
    print(f" {nameProblem}  |  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(separator)
    print(f"  displacement: {fmt_array(displacement)}")
    print(f"  reaction: {fmt_array(reaction)}")

    for name, value in others.items():
        shape_info = f"{value.shape}: " if isinstance(value, np.ndarray) else ""
        formatted = fmt_array(value) if isinstance(value, np.ndarray) else repr(value)
        print(f"  {name:<12} {shape_info}{formatted}")

    print(separator)

def vibrationModes(V, D, mask, numberNodes, nodeCoordinates, width = 120, **others):

    if True:
        os.system("cls" if os.name == "nt" else "clear")

    separator = "=" * width

    eigenvectors[np.ix_(mask, mask)] = D
    eigenvectors = eigenvectors[0:numberNodes, 0: numberNodes]

    fig, ax = plt.subplots(4, 1, figsize=(8, 12))
    for i in range(4):
        mode = eigenvectors[:, i + 1]
        ax[i].plot(nodeCoordinates, mode, marker='o')
        ax[i].grid()
        ax[i].set_ylim(-100, 100)
        ax[i].set_xlim(0, 1)
        ax[i].set_title(f"Mode {i+1}")
        ax[i].set_ylabel("Displacement")
        if i < 3:
            ax[i].tick_params(labelbottom=False)   # ← hides x tick labels

    plt.xlabel("Beam length")
    plt.tight_layout()
    plt.show()

    # Natural frequencies (rad/s)
    omega = np.sqrt(V)

    # Convert to Hz
    frequencies = omega / (2 * np.pi)

    print(separator)
    print(f" Vibration Modes  |  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(separator)
    print(f"  omega: {fmt_array(omega)}")
    print(f"  frequencies: {fmt_array(frequencies)}")


    for name, value in others.items():
        shape_info = f"{value.shape}: " if isinstance(value, np.ndarray) else ""
        formatted = fmt_array(value) if isinstance(value, np.ndarray) else repr(value)
        print(f"  {name:<12} {shape_info}{formatted}")

    print(separator)

def plateMindlinStiffness(GDOF, elementNodes, numberNodes, CShear, CBending, nodeCoordinates, dofPerNode):
    stiffness = np.zeros((GDOF, GDOF))

    points, weights = gaussPoints(2)

    XP, YP = np.meshgrid(points, points)
    XW, YW = np.meshgrid(weights, weights)

    gsPoints = np.stack((XP.flatten(), YP.flatten()), axis=1)
    gsWeights = np.stack((XW.flatten(), YW.flatten()), axis=1)

    # cycle for element
    for nodes in elementNodes:
        DOF = [nodes[0], nodes[1], nodes[2], nodes[3], 
                nodes[0] + numberNodes, nodes[1] + numberNodes, nodes[2] + numberNodes, nodes[3] + numberNodes,
                nodes[0] + 2*numberNodes, nodes[1] + 2*numberNodes, nodes[2] + 2*numberNodes, nodes[3] + 2*numberNodes]
        ndof = len(nodes)

        # cycle for Gauss point
        for p, w in zip(gsPoints, gsWeights):
            # part related to the mapping
            # shape functions and derivatives
            shapeFunction, natDerQ4 = shapeFunctionQ4(p[0],p[1])
            natDerQ4 = natDerQ4[:, :2]

            # Jacobian matrix - derivatives w.r.t. x,y
            Jacob, _ = Jacobian(nodeCoordinates[nodes], natDerQ4)
            XYDer = np.linalg.inv(Jacob) @ natDerQ4.T
            detJ = np.linalg.det(Jacob)
            
            # [B] matrix bending
            BBending = np.zeros((3, 3*ndof))
            BBending[0, ndof:2*ndof]   = XYDer[0, :]
            BBending[1, 2*ndof:3*ndof] = XYDer[1, :]
            BBending[2, ndof:2*ndof]   = XYDer[1, :]
            BBending[2, 2*ndof:3*ndof] = XYDer[0, :]

            # Stiffness matrix - bending
            stiffness[np.ix_(DOF, DOF)] += BBending.T @ CBending @ BBending * w[0]*w[1]*detJ
        

    points, weights = gaussPoints(1)

    XP, YP = np.meshgrid(points, points)
    XW, YW = np.meshgrid(weights, weights)

    gsPoints = np.stack((XP.flatten(), YP.flatten()), axis=1)
    gsWeights = np.stack((XW.flatten(), YW.flatten()), axis=1)

    # cycle for element
    for nodes in elementNodes:
        DOF = [nodes[0], nodes[1], nodes[2], nodes[3], 
                nodes[0] + numberNodes, nodes[1] + numberNodes, nodes[2] + numberNodes, nodes[3] + numberNodes,
                nodes[0] + 2*numberNodes, nodes[1] + 2*numberNodes, nodes[2] + 2*numberNodes, nodes[3] + 2*numberNodes]
        ndof = len(nodes)

        # cycle for Gauss point
        for p, w in zip(gsPoints, gsWeights):
            # part related to the mapping
            # shape functions and derivatives
            shapeFunction, natDerQ4 = shapeFunctionQ4(p[0],p[1])
            natDerQ4 = natDerQ4[:, :2]

            # Jacobian matrix - derivatives w.r.t. x,y
            Jacob, _ = Jacobian(nodeCoordinates[nodes], natDerQ4)
            XYDer = np.linalg.inv(Jacob) @ natDerQ4.T
            detJ = np.linalg.det(Jacob)

            # [B] matrix bending
            BShear = np.zeros((2, 3*ndof))
            BShear[0, 0:ndof] = XYDer[0,:]
            BShear[1, 0:ndof] = XYDer[1,:]
            BShear[0, ndof:2*ndof]   = shapeFunction.flatten()   # −θx
            BShear[1, 2*ndof:3*ndof] = shapeFunction.flatten()   # −θy
            
            # Stiffness matrix - bending
            stiffness[np.ix_(DOF, DOF)] += BShear.T @ CShear @ BShear * w[0]*w[1]*detJ

    return stiffness

def plateMindlinForce(GDof, numberElements, elementNodes, numberNodes, nodeCoordinates, P):
    """
    Computation of force vector for Mindlin plate element.

    Inputs:
        GDof           - Total number of degrees of freedom
        numberElements - Number of elements
        elementNodes   - (numElements x 4) connectivity array
        numberNodes    - Total number of nodes
        nodeCoordinates- (numNodes x 2) [x, y] node coordinates
        P              - Distributed load magnitude

    Outputs:
        force          - (GDof,) force vector
    """

    force = np.zeros(GDof)

    # Gauss quadrature (reduced integration for Mindlin)
    points, weights = gaussPoints(1)

    XP, YP = np.meshgrid(points, points)
    XW, YW = np.meshgrid(weights, weights)

    gsPoints = np.stack((XP.flatten(), YP.flatten()), axis=1)
    gsWeights = np.stack((XW.flatten(), YW.flatten()), axis=1)
    
    # cycle for element
    for nodes in elementNodes:

        # cycle for Gauss point
        for p, w in zip(gsPoints, gsWeights):
            
            # part related to the mapping
            # shape functions and derivatives
            shapeFunction, natDerQ4 = shapeFunctionQ4(p[0],p[1])
            natDerQ4 = natDerQ4[:, :2]

            # Jacobian matrix - derivatives w.r.t. x,y
            Jacob, _ = Jacobian(nodeCoordinates[nodes], natDerQ4)
            XYDer = np.linalg.inv(Jacob) @ natDerQ4.T          

            # Jacobian, inverse Jacobian, physical derivatives
            Jacob, _ = Jacobian(nodeCoordinates[nodes], natDerQ4)

            # Assemble force vector at the nodes of this element
            force[nodes] += shapeFunction.flatten() * P * np.linalg.det(Jacob) * w[0] * w[1]

    return force