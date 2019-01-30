import numpy as np
from scipy.linalg import solve, svdvals
from scipy.optimize import root, nnls
import scipy.optimize as o
from numba import njit


@njit()
def complex_delta_relations(Double_Delta, L, N, Z, g, Gamma):
    """Express the relations of the Delta parameters (Eqs 3 and 4).

    The relations have the form of a vector whose entries are zeros.
    """
    rels = np.zeros(L+1, np.complex128)
    Delta = Double_Delta[:L] + 1j * Double_Delta[L:]
    # Eq 3.
    rels[:L] = (-Delta**2 + g**2*N*(L-N)*Gamma - 2*Delta
                + g*np.sum(Z, axis=1)*Delta - g*np.dot(Z, Delta))

    # Eq 4.
    rels[L] = np.sum(Delta) + 2*N

    return np.concatenate((rels.real, rels.imag))


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


def der_delta(Delta, L, N, Z, g, Gamma, throw=False, scale=1):
    """Compute the derivatives of Delta (Eqs 5 and 6).
    If throw is true, will return an error if the linear equations
    solved to find derivatives is poorly conditioned."""
    A = scale*(np.diag(Delta + 1 - g/2*np.sum(Z, axis=1)) + g*Z/2)
    b = scale*(g*N*(L-N)*Gamma*np.ones(L) + np.sum(Z, axis=1)*Delta/2
         - np.dot(Z, Delta)/2)

    # c = b - 0.5*L*np.matmul(A,Delta)/g
    # y = np.zeros(L, np.float64)
    for i in range(L):
        A[i] = A[i] - A[i, -1]
    A = A[:-1, :-1]
    # c = c[:-1]
    b = b[:-1]

    # Compute the condition number of A. It quantifies how much
    # precision we loose when we solve the linear system Ax = b.
    w = svdvals(A)
    cn = w[-1]/w[0]
    # det = np.prod(w)
    # print('Det is {}'.format(det))
    cn = np.linalg.cond(A)

    # This is no longer accurate and I don't get it
    print('Condition number: {}'.format(np.round(cn,2)))
    print('We are losing around {} digits.'.format(np.round(np.log10(cn),2)))
    # print('The derivatives of delta are accurate'
          # + f' up to the {16 + np.log10(cn):3.0f}th digit.')
    x = np.zeros(L, np.float64)
    x[:-1] = solve(A, b)
    x[-1] = -np.sum(x[:-1])
    # print('This is at g = {}'.format(g))
    # prod = np.matmul(A, x[:-1]) - b
    # maxdif = np.max(np.abs(prod))
    # mindif = np.min(np.abs(prod))
    # print('Relative max error of Ax-b = 0 is {}'.format(mindif/maxdif))
    # print(x)

    return x/scale, A


def compute_particle_number(Delta, L, N, Z, g, Gamma):
    """Compute the occupation numbers (Eq 11)."""
    ders, A = der_delta(Delta, L, N, Z, g, Gamma)
    n = -0.5*Delta +  0.5*g*ders
    return n, A


def make_Z(L, epsilon):
    Z = np.zeros((L, L), np.complex128)
    for i in range(L):
        for j in range(i):  # j < i.
            Z[i, j] = (epsilon[i] + epsilon[j])/(epsilon[i]-epsilon[j])
            Z[j, i] = -Z[i, j]
    return Z


def compute_hyperbolic_energy(L, N, G, epsilon,
        g_step, holdover=0, taylor_expand=False,
        return_matrix=False):
    """Compute the exact energy using the integrals of motion.

    Args:
        L (int): system's size.
        N (int): particle number.
        G (float): interaction strength.
        epsilon (1darray of floats): epsilon values.

    Returns:
        E (float): ground state energy.
        n (1darray of floats): occupation numbers.

    """
    Gamma = -1 # hyperbolic case
    impart = np.cos(np.linspace(0, np.pi, L))
    cepsilon = epsilon + 1j*impart
    # Compute Z matrix.
    Z = make_Z(L, cepsilon)

    # Initial values for Delta with g small. The -2 values (initially
    # occupied states) go where the epsilons are smallest.
    delta = np.zeros(L, np.complex128)
    eps_min = np.argsort(epsilon)
    delta[eps_min[:N]] = -2

    # Finding value of g (used in numerics) corresponding to G
    # np.seterr(all='raise')
    lambd = 1/(1 + G*(N - L/2 - 1))
    g_final = -G*lambd

    # Points over which we will iterate until we reach g_final.
    if g_final > 0:
        g_path = np.append(np.arange(0, g_final, g_step), g_final)
    elif g_final < 0:
        g_path = -np.append(np.arange(0, -g_final, g_step), -g_final)
    else:
        g_path = np.array([0])
    # Finding root while varying g, using prev. solution to start
    double_delta = np.concatenate((delta.real, delta.imag))
    for g in g_path:
        if not np.isnan(g): # skipping steps where lambd broke
            sol = root(complex_delta_relations, double_delta, args=(L, N, Z, g, Gamma),
                       method='lm')
        if np.isnan(g):
            print('Division by 0 problem at g={}'.format(g))
        last = double_delta
        double_delta = (1 - holdover) * sol.x + holdover * last
    g = g_final
    if holdover != 0:
        sol = root(complex_delta_relations, double_delta, args=(L, N, Z, g, Gamma),
                method = 'lm')
        double_delta = sol.x
    # checking accuracy of solutions
    dr = complex_delta_relations(double_delta, L, N, Z, g, Gamma)
    if np.max(dr)> 10**-12:
        print('WARNING: At G= {} error is {}'.format(
                G, np.max(dr)))
        success = False
    else: # at least reasonably accurate
        success = True
    print('Before reducing impart, max diff is {}'.format(np.max(dr)))
    imscale = 1 - np.linspace(0, 1, 100)
    for i_s in imscale:
        Z = make_Z(L, epsilon + i_s*1j*impart)
        sol = root(complex_delta_relations, double_delta, args=(L, N, Z, g, Gamma),
                   method='lm')
        double_delta = sol.x
    # Now throwing away the imaginary part (badbadbad)
    delta = double_delta[:L]
    # delta = double_delta[:L] + 1j*double_delta[L:]
    # print(delta)

    # Now forming eigenvalues of IM and observables
    Z = make_Z(L, epsilon)
    ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
    E = 1/lambd*np.dot(epsilon, ri) + np.sum(epsilon)*(1/2 - 3/4*G)
    n, A = compute_particle_number(delta, L, N, Z, g, Gamma)
    if return_matrix:
        return E, n, delta, A
    return E, n, delta, success


def compute_iom_energy(L, N, G, model, epsilon,
        steps=100, return_delta=False, return_n=True,
        taylor_expand=False):
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
    Z = np.zeros((L, L), np.float64)
    for i in range(L):
        for j in range(i):  # j < i.
            if model == 'hyperbolic':
                Z[i, j] = (epsilon[i] + epsilon[j])/(epsilon[i]-epsilon[j])
            elif model == 'rational':
                Z[i, j] = 1/(epsilon[i] - epsilon[j])
            Z[j, i] = -Z[i, j]

    # Initial values for Delta with g small. The -2 values (initially
    # occupied states) go where the epsilons are smallest.
    delta = np.zeros(L, np.float64)
    eps_min = np.argsort(epsilon)
    delta[eps_min[:N]] = -2

    G_step = np.abs(G)/steps
    if G == 0:
        G_path = np.array([0])
    elif G > 0:
        G_path = np.append(np.arange(0, G, G_step), G)
    else:
        G_path = -np.append(np.arange(0, -G, G_step), -G)
    # g_step = np.abs(g_final)/steps
    # g_path = np.linspace(0, np.abs(g_final), steps)
    # if g_final > 0:
        # g_path = np.append(np.arange(0, g_final, g_step), g_final)
    # elif g_final < 0:
        # g_path = -np.append(np.arange(0, -g_final, g_step), -g_final)
    # inc = np.all(g_path[1:] >= g_path[:-1])
    # dec = np.all(g_path[1:] <= g_path[:-1])
    # finding root while varying g, using prev. solution to start

    # Finding value of g (used in numerics) corresponding to G
    if model =='rational':
        g_path = -2*G_path
    else: # hyperbolic
        lambd = 1/(1 + G_path*(N - L/2 - 1))
        g_path = -G_path*lambd

    for g in g_path:
        sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                   method='lm')
        delta = sol.x

        if taylor_expand and g != g_final:
            # Now applying taylor approx for next delta(g+dg)
            # Not doing this for final g value
            try:
                ddelta = der_delta(delta, L, N, Z, g, Gamma,
                                   throw=True)
                delta = delta + g_step*ddelta
            except Exception as e:
                print('Problem with derivatives: {}'.format(e))

    # checking accuracy of solutions
    dr = delta_relations(delta, L, N, Z, g, Gamma)
    if np.max(dr)> 10**-12:
        print('WARNING: At G= {} error is {}'.format(
                G, np.max(dr)))

    # Now forming eigenvalues of IM and observables
    if model == 'rational':
        # Eigenvalues of the IM.
        ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
        E = np.dot(epsilon, ri) + np.sum(epsilon)/2+G*((N-L/2)**2-N- L/4)
        if return_n:
            n, A = compute_particle_number(delta, L, N, Z, g, Gamma)
    elif model == 'hyperbolic':
        # Eigenvalues of the IM.
        ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
        E = 1/lambd[-1]*np.dot(epsilon, ri) + np.sum(epsilon)*(1/2 - 3/4*G)
        if return_n:
            n, A = compute_particle_number(delta, L, N, Z, g, Gamma)

    if return_delta and return_n:
        return E, n, delta, A
    elif return_n:
        return E, n
    else:
        return E

def rgk_spectrum(L, t1, t2):
    k = np.linspace(0, 1, L)*np.pi
    epsilon = -0.5 * t1 * (np.cos(k) - 1) - 0.5 * t2 * (np.cos(2*k) -1)
    return k, epsilon

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    L = 50
    N = 30
    k, epsilon = rgk_spectrum(L, 1, 0)
    Gc = 1./(L-2*N+1)
    Gs = np.linspace(1.5*Gc, 0, 15)
    energies = np.zeros(15)
    for i, G in enumerate(Gs):
        energies[i], _, _, _ = compute_hyperbolic_energy(L, N, G, epsilon, 0.01)
    plt.scatter(Gs, energies)
    plt.axvline(Gc)
    plt.show()
