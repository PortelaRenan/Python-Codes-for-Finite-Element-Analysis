import numpy as np
from FEA_functions import stiffness2Dtruss, stress2Dtruss, mesh

E = 30e6 # modulus od elasticity
A = 2    # area of cross section

numberElements = 3 # number of Elements
numberNodes = 4
elementNodes = [[0, 1],
                [0, 2],
                [0, 3]]
nodeCoordinates = [[0, 0],     # x-coordinates [:, 0]
                   [0, 120],   # y-coordinates [:, 1]
                   [120, 120],
                   [120, 0]]

# structure
GDOF = 2*numberNodes 
displacements = np.zeros(GDOF)
force = np.zeros(GDOF)

# applied load at node 2
force[1] = -10e3

# stiffness
stiffness = stiffness2Dtruss(E, A, GDOF, numberElements, elementNodes, numberNodes, nodeCoordinates)

# boundary conditions
bc = np.arange(2, 8)
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