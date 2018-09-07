import numpy as np
from scipy.linalg import solve, svdvals
from scipy.optimize import root
from numba import njit




@njit()
def delta_relations(Delta, L, N, Z, g, Gamma):
    """Express the relations of the Delta parameters (Eqs 3 and 4).

    The relations have the form of a vector whose entries are zeros.
    """
    rels = np.zeros(L+1, np.float64)

    # Eq 3.
    rels[:L] = (-Delta**2 + g**2*N*(L-N)*Gamma - 2*Delta
                + g*np.sum(Z, axis=1)*Delta - g*np.dot(Z, Delta))

    # Eq 4.
    rels[L] = np.sum(Delta) + 2*N

    return rels


def der_delta(Delta, L, N, Z, g, Gamma):
    """Compute the derivatives of Delta (Eqs 5 and 6)."""
    """Used for particle numbers mostly"""
    A = np.diag(Delta + 1 - g/2*np.sum(Z, axis=1)) + g*Z/2
    b = (g*N*(L-N)*Gamma*np.ones(L) + np.sum(Z, axis=1)*Delta/2
         - np.dot(Z, Delta)/2)

    # Solve the linear system with the modification specified in the
    # notes.
    x = np.zeros(L, np.float64)
    for i in range(L):
        A[i] = A[i] - A[i, -1]
    A = A[:-1, :-1]
    b = b[:-1]
    x[:-1] = solve(A, b)
    x[-1] = -np.sum(x[:-1])

    # Compute the condition number of A. It quantifies how much
    # precission we loose when we solve the linear system Ax = b.
    w = svdvals(A)
    cn = w[-1]/w[0]
    # print(f'Condition number: {cn:4.2e}')
    # print('The derivatives of delta are accurate in the worst case scenario'
          # + f' up to the {16 + np.log10(cn):3.0f}th digit.')

    return x


def compute_particle_number(Delta, L, N, Z, g, Gamma):
    """Compute the occupation numbers (Eq 11)."""
    n = -Delta/2 + g/2*der_delta(Delta, L, N, Z, g, Gamma)
    return n


def compute_iom_energy(L, N, G, model, epsilon, 
        # steps=10):
        G_step=0.0002, return_delta=False):
    """Compute the exact energy using the integrals of motion.

    Args:
        L (int): system's size.
        N (int): particle number.
        G (float): interaction strength.
        model (str): type of integrable model. Can be 'rational' or
            'hyperbolic'.
        epsilon (1darray of floats): epsilon values.

    Returns:
        E (float): ground state energy.
        n (1darray of floats): occupation numbers.

    """
    # Determine the value of the constant Gamma.
    if (model != 'rational') and (model != 'hyperbolic'):
        raise ValueError('Model should be either rational or hyperbolic.')

    Gamma = 0 if model == 'rational' else -1

    # Compute Z matrix.
    Z = np.zeros((L, L))
    for i in range(L):
        for j in range(i):  # j < i.
            if model == 'hyperbolic':
                Z[i, j] = (epsilon[i] + epsilon[j])/(epsilon[i]
                                                     - epsilon[j])
            elif model == 'rational':
                Z[i, j] = 1/(epsilon[i] - epsilon[j])
            Z[j, i] = -Z[i, j]

    # Initial values for Delta with g small. The -2 values (initially
    # occupied states) go where the epsilons are smallest.
    delta = np.zeros(L, np.float64)
    eps_min = np.argsort(epsilon)
    delta[eps_min[:N]] = -2

    # Points over which we will iterate until we reach G.
    if G>=0:
        G_path = np.append(np.arange(0, G, G_step), G)
    else:
        G_path = np.append(np.arange(0, -G, G_step), -G)
        G_path = -G_path
    Ginc = np.all(G_path[1:] >= G_path[:-1])
    Gdec = np.all(G_path[1:] <= G_path[:-1])
    if Ginc:
        print('G_path is increasing')
    elif Gdec:
        print('G_path is decreasing')
    else:
        print('Something is wrong with G_path')
        print(G_path)
    # if G==0:
        # G_path = np.array([0.])
    # elif G<0:
        # G_path = np.linspace(0, -G, steps)
        # G_path = -G_path
    # else:
        # G_path = np.linspace(0, G, steps)
    # print("G_path is {}".format(G_path))

    # setting up path to take while varying g
    if model == 'rational':
        # Lowercase g is the interaction strength of the equivalent
        # integrable Hamiltonian.
        g_path = -2*G_path

    elif model == 'hyperbolic':
        # Parametrize g to equivalent integrable Hamiltonian.
        lambd = 1/(1 + G_path*(N - L/2 - 1))
        g_path = -G_path*lambd

    # finding root while varying g, using prev. solution to start
    for g in g_path:
        sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma), 
                   method='lm')
        delta = sol.x

    # checking accuracy of solutions
    dr = delta_relations(delta, L, N, Z, g, Gamma)
    if np.max(dr)> 10**-12:
        print('WARNING: Largest error in quadratic solution is {}'.format(
            np.max(dr)))
    
    # Now forming eigenvalues of IM and observables
    if model == 'rational':
        # Eigenvalues of the IM.
        ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
        E = np.dot(epsilon, ri) + np.sum(epsilon)/2 + G*((N-L/2)**2 - N - L/4)
        n = compute_particle_number(delta, L, N, Z, g, Gamma)

    elif model == 'hyperbolic':
        # Eigenvalues of the IM.
        ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
        E = 1/lambd[-1]*np.dot(epsilon, ri) + np.sum(epsilon)*(1/2 - 3/4*G)
        n = compute_particle_number(delta, L, N, Z, g, Gamma)

    if return_delta:
        return E, n, delta
    else:
        return E, n


def compute_iom_energy_quad(L, N, G, A, B, C, epsilon, G_step=0.0002):
    """Compute the exact energy using the integrals of motion for
	quadratic coupling case.

    Args:
        L (int): system's size.
        N (int): particle number.
	    G (float): overall coupling strength
        A (float): strength of constant term in Z
        B (float): strength of linear term in Z
        C (float): strength of quadratic term in Z
        epsilon (1darray of floats): epsilon values.

    Returns:
        E (float): ground state energy.
        derE (float): dE/dg

    """
    # Determine the value of the constant Gamma.
    Gamma = A*C-B**2
    # if Gamma > 0:
        # print('Warning: trigonometric case, idk what happens now')
    # Form some constants that show up in a bit.
    seps = np.sum(epsilon)
    seps2 = np.sum(epsilon**2)
    M = N - L/2

    # Compute Z matrix.
    Z = np.zeros((L, L))
    for i in range(L):
        for j in range(i):  # j < i.
            numer = A + B*(epsilon[i] + epsilon[j]) + C*epsilon[i]*epsilon[j]
            Z[i, j] = numer/(epsilon[i] - epsilon[j])
            Z[j, i] = -Z[i, j]
    # Initial values for Delta with g small. The -2 values (initially
    # occupied states) go where the epsilons are smallest.
    delta = np.zeros(L, np.float64)
    eps_min = np.argsort(epsilon)
    delta[eps_min[:N]] = -2

    # Points over which we will iterate until we reach G.
    G_path = np.append(np.arange(0, G, G_step), G)

    # Parametrize g to equivalent integrable Hamiltonian \sum_i \e_i R_i
    g_scale = -2/(1 + G_path*(2*B*(M - 1) - C*seps))
    g_path = G_path*g_scale

    for g in g_path:
        sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                   method='lm')
        delta = sol.x
    g = g_path[-1]
    Lambda = 1 + B*g*(M - 1)

    # Eigenvalues of the IOM.
    ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
    # Forming energy from scaled IOM
    const = g*(3*A*L+6*B*seps+C*(seps2-seps**2))/8-A*g*M*(M-1)/2+Lambda*seps/2
    coeff = 1/(Lambda - g*C*seps/2)
    E = coeff*(np.dot(epsilon, ri) + const)
    
    # Forming dEdG
    dDelta = der_delta(delta, L, N, Z, g, Gamma)
    dri = -1/2*dDelta + 1/4*np.sum(Z, axis=1)
    dg = -2./(1+G*(2*B*(M-1)-C*seps))**2
    dcoeff = -A*M*(M-1)/2+(3*A*L+6*B*seps+C*(seps2-seps**2))/8+B*(M-1)*seps/2
    
    dEdG = dg*(C*seps*coeff**2/2 * (np.dot(epsilon, ri) + const)
            + coeff*(np.dot(epsilon, dri) + dcoeff))

    n = compute_particle_number(delta, L, N, Z, g, Gamma)
    return E, dEdG, n

if __name__ == "__main__":
    L = 50
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    N = L//2
    E, dE, n = compute_iom_energy_quad(L, N, 0.01, 1, 2, 1, epsilon)
    print('Energy is {}'.format(E))
    print('dEdG is {}'.format(dE))

