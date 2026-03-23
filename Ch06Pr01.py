import numpy as np
from FEA_functions import BernoulliBeam, BernoulliBeamBC, distributedLoad
import matplotlib.pyplot as plt

E = 1 # modulus od elasticity
I = 1    # second moment of area

# coordinates and connectivities
L = 1
numberElements = 80
nodeCoordinates = np.linspace(0,L,numberElements+1)
numberNodes = np.size(nodeCoordinates)
elementNodes = [[i, i + 1] for i in range(numberElements)]

# distributed load 
P = -1

# structure
GDOF = 2*numberNodes 
displacements = np.zeros(GDOF)
load = distributedLoad(GDOF, elementNodes, nodeCoordinates, P)

# stiffness
stiffness = BernoulliBeam(GDOF, elementNodes, nodeCoordinates, E, I)

bc = 'Clamped-clamped'
#bc = 'Clamped'
#bc = 'Simply supported'
# boundary conditions
fixedDOF = BernoulliBeamBC(bc = bc, numberElements = numberElements)
mask = np.arange(GDOF)
mask = np.bool([0 if x in fixedDOF else 1 for x in mask])


# solution
displacements[mask] = np.linalg.solve(stiffness[np.ix_(mask, mask)], 
                                       load[mask])

np.set_printoptions(precision=4)
print(f'Displacement = \n{displacements.reshape((-1, 2))} mm')

# reaction
reaction = np.matmul(stiffness, displacements)/1e3
print(f'Reaction = \n{reaction.reshape((-1, 2))} kN')

xDis = displacements.reshape((-1, 2))[:,0]
plt.plot(nodeCoordinates, xDis, linestyle = '-')
plt.show()

