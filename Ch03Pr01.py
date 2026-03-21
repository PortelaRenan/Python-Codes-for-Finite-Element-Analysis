import numpy as np
import FEA_functions as FEA

E = 30e6 # modulus od elasticity
A = 1    # area of cross section
L = 90   # length of bar

numberElements = 3 # number of Elements
nodeCoordinates = np.linspace(0, L, numberElements + 1)
numberNodes = len(nodeCoordinates)

# connection at elements
var = np.arange(numberElements, dtype=int)
elementNodes =  np.zeros([len(var), 2], dtype=int)
elementNodes[:, 0] = var
elementNodes[:, 1] = var + 1

# structure
displacement = np.zeros(numberNodes)
force = np.zeros(numberNodes)
stiffness = np.zeros([numberNodes, numberNodes])

# applied load at node 2
force[1] = 3000

# boundary conditions
is_fixed = (nodeCoordinates == np.min(nodeCoordinates)) | (nodeCoordinates == np.max(nodeCoordinates))
fixedDOF: np.array = np.ones((len(nodeCoordinates), 2)) 
fixedDOF[:, 0] = nodeCoordinates
fixedDOF[:, 1] = [not xx for xx in is_fixed]

freeDOF = np.bool(fixedDOF[:, 1])

# stiffness matrix
for e in range(numberElements):
    DOF = elementNodes[e, :]
    
    lengthElement = nodeCoordinates[DOF[1]] - nodeCoordinates[DOF[0]]
    detJac = lengthElement/2

    points, weights = FEA.gaussPoints(1)
    _, B = FEA.shapeFunctionL2(points)
    
    B_local = B/detJac
    
    stiffness[np.ix_(DOF, DOF)] += np.outer(B_local, B_local)*detJac*E*A


# solution
displacement[freeDOF] = np.linalg.solve(stiffness[np.ix_(freeDOF, freeDOF)], 
                                           force[freeDOF])

print(f'Displacement = {displacement}')

# reaction
reaction = np.matmul(stiffness, displacement)

print(f'Reaction = {reaction}')