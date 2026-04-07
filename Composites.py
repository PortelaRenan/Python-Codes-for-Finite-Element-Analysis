import numpy as np

def OrthotropicCompliance(
    E1: float, E2: float, E3: float,
    nu12: float, nu23: float, nu13: float,
    G12: float, G23: float, G13: float
) -> np.ndarray[np.float64]:
    """
    This function returns the compliance matrix for orthotropic materials. There are nine
    arguments representing the nine independent material constants. The size of the compliance
    matrix is 6 x 6.
    """

    return np.array([[1/E1, -nu12/E1, -nu13/E1, 0, 0, 0],
                    [-nu12/E1, 1/E2, -nu23/E2, 0, 0, 0],
                    [-nu13/E1, -nu23/E2, 1/E3, 0, 0, 0],
                    [0, 0, 0, 1/G23, 0, 0],
                    [0, 0, 0, 0, 1/G13, 0],
                    [0, 0, 0, 0, 0, 1/G12]])

def OrthotropicStiffness(E1: float, E2: float, E3: float,
    nu12: float, nu23: float, nu13: float,
    G12: float, G23: float, G13: float
) -> np.ndarray[np.float64]:
    """
    This function returns the stiffness matrix  for orthotropic materials. There are nine
    arguments representing the nine independent  material constants. The size of the stiffness
    matrix is 6 x 6.
    """

    return np.linalg.inv(OrthotropicCompliance(E1, E2, E3, nu12, nu23, nu13, G12, G23, G13))

def TransverselyIsotropicCompliance(
    E1: float, E2: float,
    nu12: float, nu23: float,
    G12: float
) -> np.ndarray[np.float64]:
    """
    This function returns the compliance matrix for transversely isotropic
    materials. There are five arguments representing the five independent material
    constants. The size of the compliance matrix is 6 x 6.
    """

    return np.array([[1/E1, -nu12/E1, -nu12/E1, 0, 0, 0],
                    [-nu12/E1, 1/E2, -nu23/E2, 0, 0, 0],
                    [-nu12/E1, -nu23/E2, 1/E2, 0, 0, 0],
                    [0, 0, 0, 2*(1 + nu23)/E2, 0, 0],
                    [0, 0, 0, 0, 1/G12, 0],
                    [0, 0, 0, 0, 0, 1/G12]])

def TransverselyIsotropicStiffness(
    E1: float, E2: float,
    nu12: float, nu23: float,
    G12: float
) -> np.ndarray[np.float64]:
    """
    This function returns the stiffness matrix for transversely isotropic
    materials. There are five arguments representing the five independent material
    constants. The size of the stiffness matrix is 6 x 6.
    """

    return np.linalg.inv(TransverselyIsotropicCompliance(E1,E2,nu12, nu23,G12))

def IsotropicCompliance(E: float, nu: float) -> np.ndarray[np.float64]:
    """
    This function returns the compliance matrix for isotropic materials. There are two
    arguments representing the two independent material constants. The size of the
    compliance matrix is 6 x 6.
    """

    return np.array([[1/E, -nu/E, -nu/E, 0, 0, 0],
                    [-nu/E, 1/E, -nu/E, 0, 0, 0],
                    [-nu/E, -nu/E, 1/E, 0, 0, 0],
                    [0, 0, 0, 2*(1 + nu)/E, 0, 0],
                    [0, 0, 0, 0, 2*(1 + nu)/E, 0],
                    [0, 0, 0, 0, 0, 2*(1 + nu)/E]])

def IsotropicStiffness(E: float, nu: float) -> np.ndarray[np.float64]:
    """
    This function returns the stiffness matrix for isotropic materials. There are two
    arguments representing the two independent material constants. The size of the
    stiffness matrix is 6 x 6.
    """

    return np.linalg.inv(IsotropicCompliance(E,nu))

def LongitudinalElasticModulus(Vf, E1f, Em):
    """
    This function returns Young's modulus in the longitudinal direction. 
    Its input are three values:
    Vf - fiber volume fraction
    E1f - longitudinal Young's modulus of the fiber
    Em - Young's modulus of the matrix
    """

    return Vf*E1f + (1 - Vf)*Em

def TransverseElasticModulus(Vf,E2f,Em,Eta,NU12f,NU21f,NUm,E1f,p = 1):
    """
    This function returns Young's modulus in the transverse direction. 
    Its input are nine values:
    Vf - fiber volume fraction
    E2f - transverse Young's modulus of the fiber
    Em - Young's modulus of the matrix
    Eta - stress-partitioning factor
    NU12f - Poisson's ratio NU12 of the fiber
    NU21f - Poisson's ratio NU21 of the fiber
    NUm - Poisson's ratio of the matrix
    E1f - longitudinal Young's modulus of the fiber
    p - parameter used to determine which equation to use:
        p = 1 - use equation (3.4)
        p = 2 - use equation (3.9)
        p = 3 - use equation (3.10)
    """

    if p == 1:
        return 1/(Vf/E2f + (1 - Vf)/Em)
    elif p == 2:
        return 1/((Vf/E2f + Eta*(1 - Vf)/Em)/(Vf + Eta*(1 - Vf)))
    elif p == 3:
        deno = E1f*Vf + Em*(1 - Vf)
        etaf = (E1f*Vf + ((1-NU12f*NU21f)*Em + NUm*NU21f*E1f)*(1 - Vf))/deno        
        etam = (((1-NUm*NUm)*E1f - (1-NUm*NU12f)*Em)*Vf + Em*(1 - Vf))/deno
        return 1/(etaf*Vf/E2f + etam*(1 - Vf)/Em)        
    
def ShearModulus12(Vf, G12f, Gm, EtaPrime, p = 1):
    """
    This function returns the shear modulus G12
    Its input are five values:
    Vf - fiber volume fraction
    G12f - shear modulus G12 of the fiber
    Gm - shear modulus of the matrix
    EtaPrime - shear stress-partitioning factor
    p - parameter used to determine which equation to use:
        p = 1 - use equation (3.5)
        p = 2 - use equation (3.13)
        p = 3 - use equation (3.14)
    """
    if p == 1:
        return 1/(Vf/G12f + (1 - Vf)/Gm) 
    elif p == 2:
        return 1/((Vf/G12f + EtaPrime*(1 - Vf)/Gm)/(Vf + EtaPrime*(1 - Vf)))
    elif p == 3:
        return Gm*((Gm + G12f) - Vf*(Gm - G12f))/((Gm + G12f) + Vf*(Gm - G12f))

def PoissonRatio12(Vf,nu12f,num):
    """
    This function returns Poisson's ratio NU12
    Its input are three values:
    Vf - fiber volume fraction
    nu12f - Poisson's ratio NU12 of the fiber
    num - Poisson's ratio of the matrix
    """
    return Vf*nu12f + (1 - Vf)*num

def LongitudinalThermalExpansion(Vf, E1f, Em, Alpha1f, Alpham):
    """
    This function returns the coefficient of thermal expansion in the longitudinal direction.
    Its input are five values:
    Vf - fiber volume fraction
    E1f - longitudinal Young's modulus of the fiber
    Em - Young's modulus of the matrix
    Alpha1f - coefficient of thermal expansion in the 1-direction for the fiber
    Alpham - coefficient of thermal expansion for the matrix
    """
    return (Vf*E1f*Alpha1f + (1 - Vf)*Em*Alpham)/(E1f*Vf + Em*(1 - Vf))

def TransverseThermalExpansion(Vf, Alpha2f, Alpham, E1, E1f, Em, NU1f, NUm, Alpha1f,p = 1):
    """
    This function returns the coefficient of thermal expansion in the transverse direction.
    Its input are ten values:
    Vf - fiber volume fraction
    Alpha2f - coefficient of thermal expansion in the 2-direction for the fiber
    Alpham - coefficient of thermal expansion for the matrix
    E1 - longitudinal Young's modulus of the lamina
    E1f - longitudianl Young's modulus of the fiber
    Em - Young's modulus of the matrix
    NU1f - Poisson's ratio of the fiber
    NUm - Poisson's ratio of the matrix
    Alpha1f - coefficient of thermal expansion in the 1-direction
    p - parameter used to determine which equation to use
        p = 1 - use equation (3.8)
        p = 2 - use equation (3.7)
    """

    if p == 1: 
        return Vf*Alpha2f + (1 - Vf)*Alpham
    elif p == 2:
        return (Alpha2f - (Em/E1)*NU1f*(Alpham - Alpha1f)*(1 - Vf))*Vf + (Alpham + (E1f/E1)*NUm*(Alpham - Alpha1f)*Vf)*(1 - Vf);


def ReducedCompliance(E1: float, E2: float, NU12: float, G12: float) -> np.ndarray[np.float64]:
    """
    This function returns the reduced compliance matrix for fiber-reinforced materials.
    There are four arguments representing four material constants. The size of the reduced
    compliance matrix is 3 x 3.
    """

    S = TransverselyIsotropicCompliance(E1, E2, NU12, 0, G12)
    idx = [0, 1, 5]
    return S[np.ix_(idx, idx)]

def ReducedStiffness(E1: float, E2: float, NU12: float, G12: float) -> np.ndarray[np.float64]:
    """
    This function returns the reduced isotropic compliance matrix for fiber-reinforced 
    materials. There are two arguments representing two material constants. The size of
    the reduced compliance matrix is 3 x 3.
    """
    
    return np.linalg.inv(ReducedCompliance(E1,E2, NU12, G12))

def ReducedIsotropicCompliance(E: float, nu: float) -> np.ndarray[np.float64]:
    """
    This function returns the reduced isotropic compliance matrix for fiber-reinforced 
    materials. There are two arguments representing two material constants. The size 
    of the reduced compliance matrix is 3 x 3.
    """
    S = IsotropicCompliance(E, nu)
    idx = [0, 1, 5]
    return S[np.ix_(idx, idx)]

def ReducedIsotropicStiffness(E: float, nu: float) -> np.ndarray[np.float64]:
    """
    This function returns the reduced isotropic stiffness matrix for fiber-reinforced 
    materials. There are two arguments representing two material constants. The size of
    the reduced stiffness matrix is 3 x 3.
    """

    return np.linalg.inv(ReducedIsotropicStiffness(E, nu))

def TransformationMatrix(theta: float) -> np.ndarray[np.float64]:
    """
    This function returns the transformation matrix T given the orientation angle "theta".
    There is only one argument representing "theta", which must be given in degrees.
    The size of the matrix is 3 x 3.
    """
    m = np.cos(theta*np.pi/180)
    n = np.sin(theta*np.pi/180)

    return np.array([
        [ m*m, n*n, 2*m*n],
        [ n*n, m*m, -2*m*n],
        [-m*n, m*n,  m*m-n*n]
    ])

def GlobalCoordinateStiffness(Q: np.ndarray[np.float64], theta: float) -> np.ndarray[np.float64]:  
    """
    This function returns the transformed reduced stiffness matrix given the reduced
    stiffness matrix Q and the orientation angle "theta". There are two arguments 
    representing Q and "theta".
    The size of the matrix is 3 x 3. The angle "theta" must be given in degrees.
    """
    T = TransformationMatrix(theta)
    return np.linalg.inv(T) @ Q @ T

def LaminateABD(A: np.ndarray[np.float64], B: np.ndarray[np.float64], 
                D: np.ndarray[np.float64], Qbar: np.ndarray[np.float64], 
                z: list[np.float64]):
    """
    Compute the A, B, and D stiffness matrices for a laminate.

    The laminate consists of N plies. Ply coordinates are measured from the
    laminate midplane, with z[0] at the bottom face and z[N] at the top face,
    so len(z) == len(Q_bar) + 1.

    Parameters
    ----------
    Q_bar : np.ndarray[np.float64]
        Transformed reduced stiffness matrices for each ply, each of shape
        (3, 3), ordered from bottom (ply 1) to top (ply N).
    z : list of float
        Ply interface z-coordinates, of length N+1, ordered from bottom to
        top. z[k-1] and z[k] are the bottom and top z-coordinates of ply k.

    Returns
    -------
    A : np.ndarray[np.float64]
        In-plane stiffness matrix, shape (3, 3).
    B : np.ndarray[np.float64]
        Bending-extension coupling matrix, shape (3, 3).
    D : np.ndarray[np.float64]
        Bending stiffness matrix, shape (3, 3).
    """

    dz1 = z[1] - z[0]
    dz2 = z[1] ** 2 - z[0] ** 2
    dz3 = z[1] ** 3 - z[0] ** 3

    A += Qbar * dz1
    B += Qbar * (dz2 / 2)
    D += Qbar * (dz3 / 3)

    return A, B, D

if __name__ == '__main__':
    
    np.set_printoptions(precision=4)
    
    Q = ReducedStiffness(155, 12.10, 0.248, 4.40)
    S = ReducedCompliance(155, 12.10, 0.248, 4.40)
    T = TransformationMatrix(-80)
    print(T@S@np.linalg.inv(T))