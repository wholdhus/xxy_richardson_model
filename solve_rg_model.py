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


def use_g_inv(L, N, G, Z, g_step, start=0.9, finish=1.1):
    Gamma = -1
    GP = 1./(1-N+L/2)
    lambd1 = 1/(1 + start*GP*(N - L/2 - 1))
    gf1 = -start*GP*lambd1
    g_path_1 = np.append(np.arange(0, gf1, g_step, dtype=np.float64),
                         gf1)

    g_path_1 = g_path_1[1:]
    if G < finish*GP:
        G_path_2 = -np.append(np.arange(-start*GP, -finish*GP, g_step),
                -finish*GP)
    else:
        G_path_2 = -np.append(np.arange(-start*GP, -G, g_step),
                -G)
    g_path_2 = -G_path_2/(1+G_path_2*(N-L/2-1))
    ginv_path_2 = 1/g_path_2
    # number of steps we'll take
    l = len(g_path_1) + len(ginv_path_2)

    if G < finish*GP: # still need to do the last bit
        G_path_3 = -np.append(np.arange(-finish*GP, -G, g_step), -G)
        lambds3 = 1/(1+part3G*(N-L/2-1))
        g_path_3 = -part3G*lambds3
        l = l + len(g_path_3)

    deltas = np.zeros((L,l), np.float64)
    deltas[0][:N] = -2 # assuming these have lowest epsilon

    # Now we have our route!

    for i, g in enumerate(g_path_1[1:]): # skipping 0
        delta = deltas[i]
        sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                method='lm')
        deltas[i+1] = sol.x
    Lambda = deltas[len(g_path)-1]/g_path_1[-1]

    for i, g in enumerate(ginv_path_2):
        j = i + len(g_path_1) - 1
        sol = root(lambda_relations, Lambda, args=(L, N, Z,
                   g, Gamma), method='lm')
        Lambda = sol.x
        deltas[j+1] = Lambda/g
    if G < finish*GP:
        for i, g in enumerate(part3):
            j = i + len(g_path_1) + len(ginv_path_2) - 1
            delta = deltas[j]
            sol = root(delta_relations, delta, args=(L, N, Z, g,
                       Gamma), method= 'lm')
            deltas[j+1] = sol.x
    return deltas, np.concat((g_path_1, g_path_2, g_path_3))


def make_g_path(gf, g_step):
    if gf < 0:
        g_path = -np.append(np.arange(0, -gf, g_step), -gf)
    else:
        g_path = np.append(np.arange(0, gf, g_step), gf)
    return g_path


def compute_hyperbolic_deltas(L, N, G, epsilon, g_step):
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
    if G < 0.9*Gp and try_g_inv: # need to do some trickz
        print('Using inverted g stuff')
        deltas, g_path = use_g_inv(L, N, G, Z, epsilon, g_step)
    else:
        if Grg < 0 and G < 1.1*Grg:
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
        deltas = np.zeros((len(g_path) + 1, L), np.float64)
        eps_min = np.argsort(epsilon)
        deltas[0][eps_min[:N]] = -2

        # Finding root while varying g, using prev. solution to start
        for i, g in enumerate(g_path):
            delta = deltas[i]
            sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                       method='lm')
            deltas[i+1] = sol.x
    # checking if we satisfy the delta relations
    dr = delta_relations(deltas[-1], L, N, Z, g_path[-1], Gamma)
    if np.max(dr)> 10**-12:
        print('WARNING: At G= {} error is {}'.format(
              G, np.max(dr)))
    g_path = np.concatenate(([0], g_path)) # removed 0 earlier
    return g_path, deltas, Z


def compute_hyperbolic_energy(L, N, G, epsilon, g_step):
    g_path, deltas, Z = compute_hyperbolic_deltas(L, N, G,
                                                  epsilon, g_step)
    # taking derivatives via finite difference
    # print(deltas)
    der_deltas = np.gradient(deltas, g_path, axis=0)
    l = len(g_path)
    # Now forming eigenvalues of IM and observables
    ioms = np.zeros((l, L))
    nsk = np.zeros((l, L))
    energies = np.zeros(l)
    G_path = -g_path/(1+g_path*(N-L/2-1))
    lambds = 1/(1 + G_path*(N - L/2 - 1))
    for i, g in enumerate(g_path):
        ioms[i] = -1./2 - deltas[i]/2 + g/4*np.sum(Z, axis=1)
        energies[i] = (1/lambds[i] * np.dot(epsilon, ioms[i])
                       + np.sum(epsilon)*(1./2 - 3/4*G_path[i]))
        nsk[i] = -0.5 * deltas[i] + 0.5*g*der_deltas[i]
    return energies, nsk, deltas, G_path


def rgk_spectrum(L, t1, t2):
    k = np.linspace(0, 1, L)*np.pi
    epsilon = -0.5 * t1 * (np.cos(k) - 1) - 0.5 * t2 * (np.cos(2*k) -1)
    return k, epsilon


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    g_step = float(sys.argv[3])
    G = 1.1/(L-2*N+1)
    k, epsilon = rgk_spectrum(L, 1, 0)
    E, n, Delta, G = compute_hyperbolic_energy(L, N, G, epsilon, g_step)
    plt.scatter(G, E)
    plt.axvline(1./(L-2*N+1))
    plt.show()
