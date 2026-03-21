import numpy as np


E = 70e3 # modulus od elasticity
A = 200  # area of cross section
k = 2e3  # spring stiffness

numberElements = 3 # number of Elements
numberNodes = 4
elementNodes = [[0, 1],
                [1, 2],
                [2, 3]]
nodeCoordinates = [0, 2000, 4000, 4000]

# structure
displacement = np.zeros(numberNodes)
force = np.zeros(numberNodes)
stiffness = np.zeros([numberNodes, numberNodes])

# applied load at node 2
force[1] = 8000

# stiffness matrix
for e in range(numberElements):
    DOF = elementNodes[e]
    lengthElement = nodeCoordinates[DOF[1]] - nodeCoordinates[DOF[0]]
    
    if e < 2:
        ea = E*A/lengthElement
    else:
        ea = k
    
    stiffness[np.ix_(DOF, DOF)] += ea*np.array([[1, -1],
                                                [-1, 1]])
    
# boundary conditions
bc = [0, 3]
mask = np.arange(numberNodes)
mask = np.bool([0 if x in bc else 1 for x in mask])

# solution
displacement[mask] = np.linalg.solve(stiffness[np.ix_(mask, mask)], 
                                     force[mask])

np.set_printoptions(precision=2)
print(f'Displacement = {displacement} mm')

# reaction
reaction = np.matmul(stiffness, displacement)/1e3

print(f'Reaction = {reaction} kN')