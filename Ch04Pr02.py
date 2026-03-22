import numpy as np
from FEA_functions import stiffness2Dtruss, stress2Dtruss, mesh

E = 70e3 # modulus od elasticity
A = 300  # area of cross section

elementNodes=[[0, 1],
              [0, 2],
              [1, 2], 
              [1, 3],
              [0, 3],
              [2, 3],
              [2, 5],
              [3, 4],
              [3, 5],
              [2, 4],
              [4, 5]]

nodeCoordinates=[[0, 0],
                 [0, 3000],
                 [3000, 0],
                 [3000, 3000],
                 [6000, 0],
                 [6000, 3000]]

numberNodes = np.size(nodeCoordinates,0) 
numberElements = np.size(elementNodes,0) 

# structure
GDOF = 2*numberNodes 
displacements = np.zeros(GDOF)
force = np.zeros(GDOF)

# applied load at node 2
force[3]  = -50e3
force[7]  = -100e3
force[11] = -50e3

# stiffness
stiffness = stiffness2Dtruss(E, A, GDOF, numberElements, elementNodes, numberNodes, nodeCoordinates)

# boundary conditions
bc = [0, 1, 9]
mask = np.arange(GDOF)
mask = np.bool([0 if x in bc else 1 for x in mask])

# solution
displacements[mask] = np.linalg.solve(stiffness[np.ix_(mask, mask)], 
                                       force[mask])

np.set_printoptions(precision=4)
print(f'Displacement = \n{displacements.reshape((-1, 2))} mm')

stress = stress2Dtruss(numberElements, elementNodes, nodeCoordinates, displacements, E)

np.set_printoptions(precision=2)
print(f'Stress = {stress} MPa')

# reaction
reaction = np.matmul(stiffness, displacements)/1e3
print(f'Reaction = \n{reaction.reshape((-1, 2))} kN')

# mesh
mesh(numberElements, elementNodes, nodeCoordinates, displacements)