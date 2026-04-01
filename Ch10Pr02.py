# Timoshenko beam in free vibrations
import numpy as np
import FEA_functions as FEA
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# E; modulus of elasticity
# G; shear modulus
# I: second moments of area
# L: length of beam
# thickness: thickness of beam
E=10e7
poisson = 0.30
L = 1
thickness=0.001
kapa = 5/6
rho = 1
I = thickness**3/12
A=1*thickness

# uniform pressure
P = -1
# constitutive matrix
G = E/2/(1+poisson)

C = np.array([[E*I, 0],
               [0, kapa*thickness*G]])

# mesh
numberElements = 40
nodeCoordinates= np.linspace(0, L, 
                             numberElements+1)

elementNodes = [[i, i + 1] for i in range(0, len(nodeCoordinates))]

# generation of coordinates and connectivities
numberNodes = np.size(nodeCoordinates,0)

# GDof: global number of degrees of freedom
GDOF = 2*numberNodes

# computation of the system stiffness, mass, force
stiffness = FEA.stiffnessTimoshenko(C, GDOF, elementNodes, nodeCoordinates)
mass = FEA.massMatrixTimoshenko(rho, thickness, nodeCoordinates, elementNodes, GDOF)
force = FEA.forceTimoshenko(GDOF, nodeCoordinates, elementNodes, P)

# boundary conditions 
bc = 'simply-supported'
if bc == 'simply-supported':
    fixedNodeW = np.array([0, numberNodes - 1])
    fixedNodeTX = np.array([])
elif bc == 'clamped':
    fixedNodeW = np.array([0, numberNodes - 1])
    fixedNodeTX = fixedNodeW
elif bc == 'cantilever':
    fixedNodeW = np.array([0])
    fixedNodeTX = np.array([0])

fixedDOF = np.concatenate([fixedNodeW, 
                          fixedNodeTX + numberNodes])
mask = np.arange(GDOF)
mask = [False if x in fixedDOF else True for x in mask]

# solution
displacements = np.zeros(GDOF)    
displacements[mask] = np.linalg.solve(stiffness[np.ix_(mask, mask)], 
                                       force[mask])

# Solve generalized eigenvalue problem: K u = λ M u
eigenvectors = np.zeros((GDOF, GDOF))
V, D = eigh(stiffness[np.ix_(mask, mask)], 
                 mass[np.ix_(mask, mask)])

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

FEA.vibrationModes(V, D, mask, numberNodes, nodeCoordinates)