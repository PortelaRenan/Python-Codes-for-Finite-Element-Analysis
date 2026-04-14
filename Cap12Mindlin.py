# Mindlin plate in bending
import numpy as np
import FEA_functions as FEA
import matplotlib.pyplot as plt

# isotropic material
E = 10920
poisson = 0.30
thickness = 0.1
kapa=5/6

# matrix C
# Bending part
CBending = FEA.isotropicMaterial(E, thickness, poisson)

# shear part
CShear = kapa*thickness*E/2/(1+poisson)*np.identity(2)

# load
P = -1

# mesh generation
L = 1
numberElementsX = 20
numberElementsY = numberElementsX
numberElements = numberElementsX*numberElementsY
[nodeCoordinates, elementNodes] = FEA.rectangularMesh(L,L,numberElementsX,numberElementsY,'Q4')
numberNodes = len(nodeCoordinates)


DofPerNode = 3 # number of kinematic parameters per node
GDOF = DofPerNode*len(nodeCoordinates)


typeBC = 'cccc'
freeDOF = FEA.plateBC(typeBC, nodeCoordinates, GDOF)

stiffness = FEA.plateMindlinStiffness(GDOF, elementNodes, numberNodes, CShear, CBending, nodeCoordinates, DofPerNode)
force = FEA.plateMindlinForce(GDOF, numberElements, elementNodes, numberNodes, nodeCoordinates, P)
displacements = np.zeros(GDOF)    

# solution
displacements[freeDOF] = np.linalg.solve(stiffness[np.ix_(freeDOF, freeDOF)], 
                                       force[freeDOF])



ax = plt.axes(projection='3d')
ax.plot3D(nodeCoordinates[:,1], nodeCoordinates[:,0], displacements[0:numberNodes], '.')
plt.show()
