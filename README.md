# 🧱 Finite Element Analysis in Python

A collection of Python implementations of the Finite Element Method (FEM) applied to structural mechanics problems. The repository is organized by problem type, with each module containing a self-contained solver, helper functions, and visualization tools.

## Overview
This repository provides clean, well-documented Python implementations of FEM solvers for structural analysis. Each problem is implemented with:
* Stiffness matrix assembly
* Boundary condition application
* Displacement and stress computation
* Mesh visualization (original vs. deformed)

---
Dependencies
```bash
pip install numpy matplotlib
```
Package	Version	Purpose
`numpy`	≥ 1.25	Matrix operations, linear algebra
`matplotlib`	≥ 3.5	Mesh and deformation plotting
---
Core Functions
`stiffness2Dtruss`
Assembles the global stiffness matrix for a 2D truss structure by looping over all elements, computing local stiffness matrices based on element geometry and material properties, and assembling them into the global system.
```python
stiffness2Dtruss(elasticModulus, crossSection, GDOF,
                 numberElement, elementNodes,
                 numberNodes, nodeCoordinates) -> np.ndarray
```
`stress2Dtruss`
Computes the axial stress in each truss element from the global displacement vector. Positive values indicate tension; negative values indicate compression.
```python
stress2Dtruss(numberElements, elementNodes,
              nodeCoordinates, displacement,
              elasticModulus) -> np.ndarray
```
`mesh`
Plots the original and deformed configurations of the truss. Displacements are scaled for visual clarity.
```python
mesh(numberElements, elementNodes,
     nodeCoordinates, displacement) -> None
```
> 🔵 Blue = Undeformed structure &nbsp;|&nbsp; 🔴 Red dashed = Deformed structure
---

📘 Chapter 3 — Analysis of Bars
> Finite element formulation and analysis of 1D bar (rod) elements under axial loading.
#	Problem	Description	File
1	Add problem name	Add brief description	`bars/examples/problem_01.py`
2	Add problem name	Add brief description	`bars/examples/problem_02.py`
---
📗 Chapter 4 — 2D Trusses
> Assembly and solution of 2D truss structures using the direct stiffness method.
#	Problem	Description	File
1	Add problem name	Add brief description	`truss/examples/problem_01.py`
2	Add problem name	Add brief description	`truss/examples/problem_02.py`
