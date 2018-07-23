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
    #       + f' up to the {16 + np.log10(cn):3.0f}th digit.')

    return x


def compute_particle_number(Delta, L, N, Z, g, Gamma):
    """Compute the occupation numbers (Eq 11)."""
    n = -Delta/2 + g/2*der_delta(Delta, L, N, Z, g, Gamma)
    return n


def compute_iom_energy(L, N, G, model, epsilon):
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
    G_step = 0.0002
    G_path = np.append(np.arange(0, G, G_step), G)
    if model == 'rational':
        # Lowercase g is the interaction strength of the equivalent
        # integrable Hamiltonian.
        g_path = -2*G_path
        for g in g_path:
            sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                       method='lm')
            delta = sol.x

        # Eigenvalues of the IM.
        ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
        E = np.dot(epsilon, ri) + np.sum(epsilon)/2 + G*((N-L/2)**2 - N - L/4)
        n = compute_particle_number(delta, L, N, Z, g, Gamma)

    elif model == 'hyperbolic':
        # Parametrize g to equivalent integrable Hamiltonian.
        lambd = 1/(1 + G_path*(N - L/2 - 1))
        g_path = -G_path*lambd
        for g in g_path:
            sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma), 
                   method='lm')
            delta = sol.x

        # Eigenvalues of the IM.
        ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
        E = 1/lambd[-1]*np.dot(epsilon, ri) + np.sum(epsilon)*(1/2 - 3/4*G)
        n = compute_particle_number(delta, L, N, Z, g, Gamma)

    return E, n


def compute_iom_energy_quad(L, N, G, A, B, C, epsilon):
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
    if Gamma > 0:
        print('Warning: trigonometric case, idk what happens now')
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
    G_step = 0.0002
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
    ni = compute_particle_number(delta, L, N, Z, g, Gamma)
    derE = 0
    # for i in range(L):
    #     derE = derE + epsilon[i]*(ri[i]-ni[i]+0.5)/g
    return E, derE
