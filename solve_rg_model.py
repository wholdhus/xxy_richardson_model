import numpy as np
from scipy.linalg import solve, svdvals
from scipy.optimize import root
# from numba import njit


# @njit()
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

    print('Condition number: {}. We lose {} digits'.format(np.round(cn,2), np.round(np.log10(cn),2)))
    # print('The derivatives of delta are accurate'
          # + f' up to the {16 + np.log10(cn):3.0f}th digit.')
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


def make_g_path(gf, g_step):
    if gf < 0:
        g_path = -np.append(np.arange(0, -gf, g_step), -gf)
    else:
        g_path = np.append(np.arange(0, gf, g_step), gf)
    return g_path


def use_g_inv(L, N, G, Z, g_step, start=0.9):
    finish = 2 - start
    # finish = 1.1
    Gamma = -1
    GP = 1./(1-N+L/2)
    lambd1 = 1/(1 + start*GP*(N - L/2 - 1))
    gf1 = -start*GP*lambd1
    g_path_1 = make_g_path(gf1, g_step)[1:]
    if G < finish*GP:
        G_path_2 = -np.append(np.arange(-start*GP + g_step/10, -finish*GP,
                              g_step/10), -finish*GP)
    else:
        G_path_2 = -np.append(np.arange(-start*GP + g_step/10, -G,
                              g_step/10),
                              -G)
    g_path_2 = -G_path_2/(1+G_path_2*(N-L/2-1))
    ginv_path_2 = 1/g_path_2
    # number of steps we'll take
    l = len(g_path_1) + len(ginv_path_2) + 1
    g_path = np.concatenate((g_path_1, g_path_2))
    if G < finish*GP: # still need to do the last bit
        G_path_3 = -np.append(np.arange(-finish*GP + g_step/10, -G,
                              g_step), -G)
        g_path_3 = -G_path_3/(1+G_path_3*(N-L/2-1))
        g_path = np.concatenate((g_path, g_path_3))

    deltas = np.zeros((l,L), np.float64)
    deltas[0][:N] = -2 # assuming these have lowest epsilon

    # Now we have our route!

    for i, g in enumerate(g_path_1): # skipping 0
        delta = deltas[i]
        sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                method='lm')
        deltas[i+1] = sol.x
    Lambda = deltas[len(g_path_1)-1]/g_path_1[-1]

    for i, g in enumerate(ginv_path_2):
        j = i + len(g_path_1)
        sol = root(lambda_relations, Lambda, args=(L, N, Z,
                   g, Gamma), method='lm')
        Lambda = sol.x
        deltas[j+1] = Lambda/g
    if G < finish*GP:
        for i, g in enumerate(g_path_3):
            j = i + len(g_path_1) + len(ginv_path_2)
            delta = deltas[j]
            sol = root(delta_relations, delta, args=(L, N, Z, g,
                       Gamma), method= 'lm')
            deltas[j+1] = sol.x
    return deltas, g_path


def compute_hyperbolic_deltas(L, N, G, epsilon, g_step, skip_Grg=False,
                              start=0.9):
    Gamma = -1 # hyperbolic case
    # Compute Z matrix.
    Z = np.zeros((L, L))
    for i in range(L):
        for j in range(i):  # j < i.
            Z[i, j] = (epsilon[i] + epsilon[j])/(epsilon[i]-epsilon[j])
            Z[j, i] = -Z[i, j]
    if 1-N+L/2 != 0:
        Gp = 1./(1-N+L/2)
    else:
        Gp = np.nan
    Grg = 1./(L-2*N+1)
    lambd = 1/(1 + G*(N - L/2 - 1))
    g_final = -G*lambd
    # print(g_final)
    if L != 2*N:
        grg = -Grg/(1+Grg*(N-L/2-1))
    else:
        grg = np.nan
    if G < start*Gp < 0: # need to do some trickz
        print('Using inverted g stuff')
        deltas, g_path = use_g_inv(L, N, G, Z, g_step, start=start)
    elif G > start*Gp > 0: # need to do similar trixkcx
        print('This is not going to work right now. Woops')
    else:
        if skip_Grg and G < Grg < 0:
            g_path_1 = make_g_path(0.95*grg, g_step)
            g_path_2 = np.append(np.arange(1.005*grg, g_final, g_step),
                                 g_final)
            g_path = np.concatenate((g_path_1, g_path_2))[1:]
        elif G > 0:
            G_path = np.append(np.arange(0, G, g_step), G)
            g_path = -G_path/(1+G_path*(N-L/2-1))
            g_path = g_path[1:]
            print(g_path)
        else:
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


def compute_hyperbolic_energy(L, N, G, epsilon, g_step, skip_Grg=False,
                              start=0.8, use_fd=True):
    g_path, deltas, Z = compute_hyperbolic_deltas(L, N, G,
                                                  epsilon, g_step,
                                                  skip_Grg=skip_Grg,
                                                  start=start)
    if use_fd:
        # taking derivatives via finite difference
        try:
            der_deltas = np.gradient(deltas, g_path, axis=0)
        except: # Need to do my own version of gradient because doesn't work on karst
            print('Numpy gradient failed. Doing my own version')
            s = np.shape(deltas)
            der_deltas = np.zeros(s)
            der_deltas[0] = (deltas[1] - deltas[0])/(g_path[1] - g_path[0])
            der_deltas[-1] = (deltas[-1] - deltas[-2])/(g_path[-1] - g_path[-2])
            for i, g in enumerate(g_path):
                if i !=0 and i != len(g_path) - 1:
                    der_deltas[i] = (deltas[i+1] - deltas[i-1])/(g_path[i+1] - g_path[i-1])
    else: # taking derivative analytically
        s = np.shape(deltas)
        der_deltas = np.zeros(s)
        for i, g in enumerate(g_path):
            der_deltas[i], _ = der_delta(deltas[i], L, N, Z, g, -1)
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
    return energies, nsk, deltas, G_path, Z


def rgk_spectrum(L, t1, t2, start_neg=True):
    r = np.array([(i+1.0)/L for i in range(L)], dtype=np.float64)
    if not start_neg:
        k1 = np.append(
                    np.array([2*(i+1)*np.pi/L for i in range(int(L/2-1))],
                    np.float64), 0)
        k2 = np.append(
                    np.array([-2*(i+1)*np.pi/L for i in range(int(L/2-1))],
                    np.float64), -np.pi)
        k = np.sort(np.concatenate((k2,k1)))
    else: # antiperiodic bc
        k1 = np.array(
                [(2*i+1)*np.pi/L for i in range(int(L/2))],
                np.float64)
        k2 = np.array(
                [-(2*i+1)*np.pi/L for i in range(int(L/2))],
                np.float64)
        k = np.sort(np.concatenate((k2, k1)))
    print(k)

    eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2)**2))
    epsilon = eta**2
    print(epsilon)
    return k, epsilon


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    g_step = float(sys.argv[3])
    G = 1.1/(L-2*N+1)
    # G = 2.0/(1-N+L/2)
    k, epsilon_rgk = rgk_spectrum(L, 1, 0)
    # epsilon = k**2
    epsilon = k**3
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    # E, n, Delta, Gs = compute_hyperbolic_energy(L, N, G, epsilon, g_step)
    # plt.scatter(Gs, E, s=8)
    E2, n2, Delta2, G2, Z = compute_hyperbolic_energy(L, N, G, epsilon, g_step,
            start=0.9, use_fd=False)
    plt.scatter(G2, E2, s=2)
    plt.axvline(1./(L-2*N+1), color='g')
    # plt.axvline(2./(L-2*N+1), color='m')
    # plt.axvline(-2./(N-1), color = 'y')
    # plt.ylim(275, 285)

    # plt.subplot(3,1,2)
    # de = np.gradient(E2, G2)
    # d2e = np.gradient(de, G2)
    # d3e = np.gradient(d2e, G2)
    # plt.scatter(G2[10:-10], d3e[10:-10])

    plt.subplot(2,1,2)
    nsN = np.array([ns[N-1] for ns in n2])
    nsN1 = np.array([ns[N] for ns in n2])
    qp = nsN - nsN1
    plt.scatter(G2, qp, s=2)
    plt.scatter(G2, nsN, s=2)
    plt.scatter(G2, nsN1, s=2)
    plt.axvline(1./(L-2*N+1), color = 'g')
    plt.axvline(1./(L-N+1), color = 'r')
    # plt.axvline(-2./(N-1), color = 'y')
    # plt.axvline(2./(L-2*N+1), color = 'm')
    plt.ylim(0, 1)
    plt.show()
