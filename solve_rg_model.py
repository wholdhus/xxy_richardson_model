import numpy as np
from scipy.linalg import solve, svdvals
from scipy.optimize import root, nnls
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


def lambda_relations(Lambda, L, N, Z, ginv, Gamma):
    rels = np.zeros(L+1, np.float64)
    ginv2 = ginv**2
    # Eq 3.
    rels[:L] = (-Lambda**2 + N*(L-N)*Gamma - 2*ginv*Lambda
                + np.sum(Z, axis=1)*Lambda - np.dot(Z, Lambda))
    # Eq 4.
    rels[L] = np.sum(Lambda) + 2*N*ginv

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
    print('We are losing around {} digits.'.format(
          np.round(np.log10(cn),2)))
    x = np.zeros(L, np.float64)
    x[:-1] = solve(A, b)
    x[-1] = -np.sum(x[:-1])
    return x/scale, A


def compute_particle_number_fd(Delta, lastDelta, nextDelta, g, g_step):
    der_delta = (nextDelta - lastDelta)/(2*g_step)
    n = -0.5 * Delta + 0.5*g*der_delta
    return n, der_delta


def compute_particle_number(Delta, L, N, Z, g, Gamma):
    """Compute the occupation numbers (Eq 11)."""
    ders, A = der_delta(Delta, L, N, Z, g, Gamma)
    n = -0.5*Delta +  0.5*g*ders
    return n, A, ders


def use_g_inv(L, N, G, Z, delta, g_step, start=0.9, finish=1.1):
    Gamma = -1
    GP = 1./(1-N+L/2)
    lambd1 = 1/(1 + start*GP*(N - L/2 - 1))
    gf1 = -start*GP*lambd1
    part1 = np.append(np.arange(0, gf1, g_step, dtype=np.float64),
            gf1)
    for g in part1:
        sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                method='lm')
        delta = sol.x # not doing any holdover stuff since it doesn't work
    if G < finish*GP:
        part2G = -np.append(np.arange(-start*GP, -finish*GP, g_step),
                -finish*GP)
    else:
        part2G = -np.append(np.arange(-start*GP, -G, g_step),
                -G)
    # print('G for part 2: {}'.format(part2G))
    part2 = -(1+part2G*(N-L/2-1))/part2G # array of g^-1
    # print('g^-1 for part 2: {}'.format(part2))
    Lambda = delta/part1[-1]
    for g in part2:
        sol = root(lambda_relations, Lambda, args=(L, N, Z,
                   g, Gamma), method='lm')
        Lambda = sol.x
    delta = Lambda/part2[-1]
    if G < finish*GP: # still need to do the last bit
        part3G = -np.append(np.arange(-finish*GP, -G, g_step), -G)
        lambds3 = 1/(1+part3G*(N-L/2-1))
        part3 = -part3G*lambds3
        for g in part3:
            sol = root(delta_relations, delta, args=(L, N, Z, g,
                       Gamma), method= 'lm')
            delta = sol.x
    return delta


def make_g_path(gf, g_step):
    if gf < 0:
        g_path = -np.append(np.arange(0, -gf, g_step), -gf)
    else:
        g_path = np.append(np.arange(0, gf, g_step), gf)
    return g_path


def compute_hyperbolic_deltas(L, N, G, epsilon,
        g_step, holdover=0, taylor_expand=False,
        try_g_inv=True, skip_Grg=True, start=0.9,
        use_finite_diff=False):
    Gamma = -1 # hyperbolic case
    # Compute Z matrix.
    Z = np.zeros((L, L))
    for i in range(L):
        for j in range(i):  # j < i.
            Z[i, j] = (epsilon[i] + epsilon[j])/(epsilon[i]-epsilon[j])
            Z[j, i] = -Z[i, j]
    Gp = 1./(1-N+L/2)
    Grg = 1./(L-2*N+1)
    lambd = 1/(1 + G*(N - L/2 - 1))
    g_final = -G*lambd
    grg = -Grg/(1+Grg*(N-L/2-1))
    if G < start*Gp and try_g_inv: # need to do some trickz
        print('Using inverted g stuff')
        deltas = use_g_inv(L, N, G, Z, epsilon, g_step, start)
    else:
        if Grg < 0 and G < 1.1*Grg and skip_Grg:
            print('Skipping Grg')
            # TODO: add holdover to cover just the jump from this
            # we hit a problem ONLY at Grg
            part1 = make_g_path(0.9*grg, g_step)
            # g is positive here even though G < 0
            part2 = np.append(np.arange(1.1*grg, g_final, g_step),
                               g_final)
            g_path = np.concatenate((part1, part2))
        else:
            # no spcial handling needed
            g_path = make_g_path(g_final, g_step)
        g_path = g_path[1:] # don't need to start at 0
        # Initial values for Delta with g small. The -2 values (initially
        # occupied states) go where the epsilons are smallest.
        deltas = np.zeros((L, len(g_path) + 1), np.float64)
        eps_min = np.argsort(epsilon)
        deltas[0][eps_min[:N]] = -2

        # Finding root while varying g, using prev. solution to start
        for i, g in enumerate(g_path):
            sol = root(delta_relations, deltas[i], args=(L, N, Z, g, Gamma),
                       method='lm')
            deltas[i+1] = (1 - holdover) * sol.x + holdover * deltas[i]
        if holdover != 0:
            sol = root(delta_relations, deltas[-1], args=(L, N, Z, g_final,
                       Gamma), method = 'lm')
            deltas[-1] = sol.x
    g = g_final
    # checking if we satisfy the delta relations
    dr = delta_relations(deltas[-1], L, N, Z, g, Gamma)
    if np.max(dr)> 10**-12:
        print('WARNING: At G= {} error is {}'.format(
                G, np.max(dr)))
    return g_path, deltas, Z


def compute_hyperbolic_energy(L, N, G, epsilon, g_step)
    g_path, deltas, Z = compute_hyperbolic_deltas(L, N, G,
                                                  epsilon, g_step)
    # taking derivatives via finite difference
    differences = np.array([g_step for i in range(len(g_path))])
    differences[-1] = g_path[-1] - g_path[-2]
    der_deltas = np.gradient(deltas, differences, axis=0)
    l = len(g_path)
    # Now forming eigenvalues of IM and observables
    ioms = np.zeros((L, l))
    energies = np.zeros(l)
    G_path = -g_path/(1+g_path(N-L/2-1))
    lambds = 1/(1 + G_path*(N - L/2 - 1))
    for i, g in enumerate(g_path):
        ioms[i] = -1./2 - delta/2 + g/4*np.sum(Z, axis=1)
        energies[i] = (1/lambds[i] * np.dot(epsilon, ioms[i])
                       + np.sum(epsilon)*(1./2 - 3/4*G_path[i]))
    return energies, nsk, deltas


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
            n, A, deriv = compute_particle_number(delta, L, N, Z, g, Gamma)
    elif model == 'hyperbolic':
        # Eigenvalues of the IM.
        ri = -1/2 - delta/2 + g/4*np.sum(Z, axis=1)
        E = 1/lambd[-1]*np.dot(epsilon, ri) + np.sum(epsilon)*(1/2 - 3/4*G)
        if return_n:
            n, A, deriv = compute_particle_number(delta, L, N, Z, g, Gamma)

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
