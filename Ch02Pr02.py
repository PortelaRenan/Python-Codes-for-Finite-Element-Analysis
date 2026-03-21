from dataclasses import dataclass
import numpy as np

@dataclass
class System:
    elementNodes: list[list] 
    GDOF: float
    numberElements: int = None
    stiffness: np.ndarray = None
    displacement: np.ndarray = None
    force: np.ndarray = None
    prescribedDOF: list = None
    freeDOF: list = None
    
system = System(elementNodes=[[0, 1], [1, 2], [1, 3]],
                GDOF=4)

system.numberElements = len(system.elementNodes)
system.displacement = np.zeros(system.GDOF)
system.force = np.zeros(system.GDOF)
system.stiffness = np.zeros([system.GDOF, system.GDOF])

system.force[1] = 10

for element in range(system.numberElements):
    DOF = system.elementNodes[element]
    system.stiffness[np.ix_(DOF, DOF)] += np.array([[1, -1], [-1, 1]])

system.prescribedDOF = [0, 2, 3]
freeDOF = np.setdiff1d(np.arange(system.GDOF), system.prescribedDOF)

system.displacement[freeDOF] = np.linalg.solve(system.stiffness[np.ix_(freeDOF, freeDOF)],
                                                      system.force[freeDOF])


print(f'Displacements = \n {system.displacement} \n')
reactionForces = np.matmul(system.stiffness, system.displacement)
print(f'Reaction forces = \n {reactionForces[np.ix_(system.prescribedDOF)]}')