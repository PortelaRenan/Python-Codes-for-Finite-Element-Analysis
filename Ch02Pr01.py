# Chapter 02 Problem 01 from MATLAB codes for Finite Element Analysis
import numpy as np
import matplotlib.pyplot as plt

# connection at elements
elementNodes = [[0, 1], [1, 2], [1, 3]]

# number of elements
numberElements = len(elementNodes)

# number of nodes
numberNodes = 4

# displacement vector
displacements = np.zeros([numberNodes, 1])

# force vector
force = np.zeros(numberNodes)

# stiffness matrix
stiffness = np.zeros([numberNodes, numberNodes])

# apply load at Node 2
force[1] = 10

# computation of the system stiffness matrix
for element in range(numberElements):
    DOF = elementNodes[element]
    stiffness[np.ix_(DOF, DOF)] += np.array([[1, -1], [-1, 1]])

# boundary conditions and solution
prescribedDOF = [0, 2, 3]
freeDOF = np.setdiff1d(np.arange(numberNodes), prescribedDOF)

# solution
displacements[freeDOF] = np.linalg.solve(stiffness[np.ix_(freeDOF, freeDOF)], 
                                                 force[freeDOF])

print(f'Displacements = \n {displacements} \n')
# reaction forces
reactionForces = np.matmul(stiffness, displacements)
print(f'Reaction forces = \n {reactionForces[np.ix_(prescribedDOF)]}')
