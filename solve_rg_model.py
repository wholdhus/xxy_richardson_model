import numpy as np
from scipy.linalg import solve, svdvals
from scipy.optimize import root
import scipy.optimize as o
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


def delta_rels_fixed(Delta, L, N, thingies, g, Gamma, epsilon):
    """Express the relations of the Delta parameters (Eqs 3 and 4)
    Combined the info from 4 into 3.
    """
    rels = np.zeros(L, np.float64)
    rels[:L] = (-Delta**2 + g**2*N*(L-N)*Gamma -2*g*N
            - (2+g*L)*Delta)
    rels[:L] = rels[:L] + 2*g*epsilon*(
                np.sum(thingies, axis=1)*Delta
                - np.dot(thingies, Delta))
    # rels[L] = np.sum(Delta) + 2*N
    print(rels)
    return rels


def der_delta(Delta, L, N, Z, g, Gamma, throw=False):
    """Compute the derivatives of Delta (Eqs 5 and 6).
    If throw is true, will return an error if the linear equations
    solved to find derivatives is poorly conditioned."""
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

    # Compute the condition number of A. It quantifies how much
    # precision we loose when we solve the linear system Ax = b.
    # w = svdvals(A)
    # cn = w[-1]/w[0]
    cn = np.linalg.cond(A)

    # print(f'Condition number: {cn:4.2e}')
    # print('The derivatives of delta are accurate'
          # + f' up to the {16 + np.log10(cn):3.0f}th digit.')
    if cn > 10**10 and throw:
        raise Exception('Condition number is {}!'.format(cn))
    else:
        x[:-1] = solve(A, b)
        x[-1] = -np.sum(x[:-1])

        return x


def compute_particle_number(Delta, L, N, Z, g, Gamma):
    """Compute the occupation numbers (Eq 11)."""
    n = -Delta/2 + g/2*der_delta(Delta, L, N, Z, g, Gamma)
    return n


def compute_hyperbolic_energy(L, N, G, epsilon,
        g_step, holdover=0, taylor_expand=False):
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

    # Compute Z matrix.
    Z = np.zeros((L, L))
    for i in range(L):
        for j in range(i):  # j < i.
            Z[i, j] = (epsilon[i] + epsilon[j])/(epsilon[i]-epsilon[j])
            Z[j, i] = -Z[i, j]

    # Initial values for Delta with g small. The -2 values (initially
    # occupied states) go where the epsilons are smallest.
    delta = np.zeros(L, np.float64)
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
    for g in g_path:
        if not np.isnan(g): # skipping steps where lambd broke
            sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                       method='lm')
        if np.isnan(g):
            print('Division by 0 problem at g={}'.format(g))
        last = delta
        delta = (1 - holdover) * sol.x + holdover * last
    g = g_final
    if holdover != 0:
        sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                method = 'lm')
        delta = sol.x
    # checking accuracy of solutions
    dr = delta_relations(delta, L, N, Z, g, Gamma)
    if np.max(dr)> 10**-12:
        print('WARNING: At G= {} error is {}'.format(
                G, np.max(dr)))
        success = False
    else: # at least reasonably accurate
        success = True

    # Now forming eigenvalues of IM and observables
    ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
    E = 1/lambd*np.dot(epsilon, ri) + np.sum(epsilon)*(1/2 - 3/4*G)
    n = compute_particle_number(delta, L, N, Z, g, Gamma)
    return E, n, dr, success


def compute_iom_energy(L, N, G, model, epsilon,
        steps=100, return_delta=False, return_n=True,
        taylor_expand=True, use_fixed_rels=False):
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
                Z[i, j] = (epsilon[i] + epsilon[j])/(epsilon[i]-epsilon[j])
            elif model == 'rational':
                Z[i, j] = 1/(epsilon[i] - epsilon[j])
            Z[j, i] = -Z[i, j]

    # Initial values for Delta with g small. The -2 values (initially
    # occupied states) go where the epsilons are smallest.
    delta = np.zeros(L, np.float64)
    eps_min = np.argsort(epsilon)
    delta[eps_min[:N]] = -2

    # Finding value of g (used in numerics) corresponding to G
    if model =='rational':
        g_final= -2*G
    else: # hyperbolic
        np.seterr(all='raise')
        lambd = 1/(1 + G*(N - L/2 - 1))
        g_final = -G*lambd

    # Points over which we will iterate until we reach g_final.
    g_step = np.abs(g_final)/steps
    if g_step > 0.02:
        print('Warning: steps are getting biggish {}'.format(
            g_step))
    # print('Step size is {}'.format(g_step))
    g_path = np.linspace(0, np.abs(g_final), steps)
    if g_final > 0:
        g_path = np.append(np.arange(0, g_final, g_step), g_final)
    elif g_final < 0:
        g_path = -np.append(np.arange(0, -g_final, g_step), -g_final)
    inc = np.all(g_path[1:] >= g_path[:-1])
    dec = np.all(g_path[1:] <= g_path[:-1])
    if inc:
        # print('g_path is increasing')
        pass
    elif dec:
        # print('g_path is decreasing')
        pass
    else:
        print('Something is wrong with g_path')
        print(g_path)
    # print(g_path)
    # finding root while varying g, using prev. solution to start
    for g in g_path:
        if not np.isnan(g): # skipping steps where lambd broke
            if use_fixed_rels:
                thingies = np.zeros((L,L))
                for i in range(L):
                    for j in range(i):
                        thingies[i, j] = 1./(epsilon[i] - epsilon[j])
                        thingies[j, i] = -thingies[i, j]
                sol = root(delta_rels_fixed, delta, args=(L, N, thingies,
                            g, Gamma, epsilon),method='lm')
            else:
                sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                           method='lm')
        if np.isnan(g):
            print('Division by 0 problem at g={}'.format(g))
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
    if use_fixed_rels:
        dr = delta_rels_fixed(delta, L, N, thingies,
                g, Gamma, epsilon)
    else:
        dr = delta_relations(delta, L, N, Z, g, Gamma)
    if np.max(dr)> 10**-12:
        print('WARNING: At G= {} error is {}'.format(
                G, np.max(dr)))
        # relerror = dr[:(L-1)]/delta
        # print('Average relative error is {}'.format(np.max(relerror)))
        # print('g_path was:')
        # print(g_path)
        # print('Differences are:')
        # print(dr)
        # print('Max delta is currently:')
        # print(np.max(np.abs(delta)))

    # Now forming eigenvalues of IM and observables
    if model == 'rational':
        # Eigenvalues of the IM.
        ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
        E = np.dot(epsilon, ri) + np.sum(epsilon)/2+G*((N-L/2)**2-N- L/4)
        if return_n:
            n = compute_particle_number(delta, L, N, Z, g, Gamma)

    elif model == 'hyperbolic':
        # Eigenvalues of the IM.
        ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
        E = 1/lambd*np.dot(epsilon, ri) + np.sum(epsilon)*(1/2 - 3/4*G)
        if return_n:
            n = compute_particle_number(delta, L, N, Z, g, Gamma)

    # print('Max delta is {}'.format(np.max(np.abs(delta))))
    # print('This is at index {}'.format(np.argmax(np.abs(delta))))
    if return_delta and return_n:
        return E, n, delta
    elif return_n:
        return E, n
    else:
        return E


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

