# Kirchhoff plate in bending
import numpy as np
import FEA_functions as FEA

# isotropic material
E = 10920
poisson = 0.30
thickness = 0.01

# load
P = -1

# 3: non-conforming 4 node element
# 4: conforming 4 node element
dof_per_node = 3 # number of kinematic parameters per node

# mesh generation
L = 1
numberElementsX = 20
numberElementsY = numberElementsX
numberElements = numberElementsX*numberElementsY


[nodeCoordinates, elementNodes] = FEA.rectangularMesh(L,L,numberElementsX,numberElementsY,'Q4')

#FEA.drawingMesh(nodeCoordinates, elementNodes, 'Q4')

GDOF = dof_per_node*len(nodeCoordinates)

typeBC = 'ssss'
freeDOF = FEA.plateBC(typeBC, nodeCoordinates, GDOF)
