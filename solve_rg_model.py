import numpy as np
from scipy.linalg import solve, svdvals
from scipy.optimize import root
# from numba import njit

TOL = 10**-10

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
    """Same as delta_relations but for lambda = delta/g,
    written in terms of ginv = g^-1
    """
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
        g_path = -np.arange(0, -gf, g_step)
    else:
        g_path = np.arange(0, gf, g_step)
    # was running into issues where this was less than machine epsilon
    if np.abs(g_path[-1] - gf) > 10 **-10:
        g_path = np.append(g_path, gf)
    else:
        g_path[-1] = gf
    return g_path


def use_g_inv(L, N, G, Z, g_step, start=0.9):
    finish = 2 - start
    Gamma = -1
    GP = 1./(1-N+L/2)
    lambd1 = 1/(1 + start*GP*(N - L/2 - 1))
    gf1 = -start*GP*lambd1
    gp1 = make_g_path(gf1, g_step)
    l1 = len(gp1)
    if G < finish*GP:
        Gp2 = -np.append(np.arange(-start*GP + g_step/10, -finish*GP,
                              g_step/10), -finish*GP)
    else:
        Gp2 = -np.append(np.arange(-start*GP + g_step/10, -G,
                              g_step/10), -G)
    gp2 = -Gp2/(1+Gp2*(N-L/2-1))
    l2 = len(gp2)
    gip2 = 1./gp2
    # number of steps we'll take
    l = l1 + l2
    g_path = np.concatenate((gp1, gp2))
    if G < finish*GP: # still need to do the last bit
        Gp3 = -np.append(np.arange(-finish*GP + g_step/10, -G,
                              g_step/5), -G)
        gp3 = -Gp3/(1+Gp3*(N-L/2-1))
        l3 = len(gp3)
        g_path = np.concatenate((g_path, gp3))
        l = l + l3
    deltas = np.zeros((l,L), np.float64)
    deltas[0][:N] = -2 # assuming these have lowest epsilon

    # Now we have our route!

    for i, g in enumerate(gp1[1:]): # we already have the g=0 solution
        delta = deltas[i]
        if i > 1: # tacking on Ddelta/Dg * Dg
            corr = (deltas[i] - deltas[i-1])*(gp1[i+1] - gp1[i])/(gp1[i] - gp1[i-1])
            delta = delta + corr
        sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                method='lm', options={'xtol':TOL})
        deltas[i+1] = sol.x

    for i, gi in enumerate(gip2):
        j = i + l1 - 1
        if i != 0 and i != 1:
            corr = (deltas[j]*gip2[i-1] - deltas[j-1]*gip2[i-2])*(gip2[i] - gip2[i-1])/(
                    gip2[i-1] - gip2[i-2])
            Lambda = deltas[j] * gip2[i-1]
        else:
            corr = (deltas[j]/g_path[j] - deltas[j-1]/g_path[j-1])*(gi - 1./g_path[j])/(
                    1./g_path[j] - 1./g_path[j-1])
            Lambda = deltas[j] / gp1[-1]
        sol = root(lambda_relations, Lambda, args=(L, N, Z, gi, Gamma),
                method='lm', options={'xtol':TOL})
        deltas[j+1] = sol.x*g_path[j+1]

    if G < finish*GP:
        for i, g in enumerate(gp3):
            j = i + l1 + l2 - 1
            corr = (deltas[j] - deltas[j-1])*(g_path[j+1] - g_path[j])/(
                    g_path[j] - g_path[j-1])
            delta = deltas[j] + corr
            sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                       method='lm', options={'xtol':TOL})
            deltas[j+1] = sol.x
    return deltas, g_path


def compute_hyperbolic_deltas(L, N, G, epsilon, g_step,
                              start=0.9, Gisg=False):
    Gamma = -1 # hyperbolic case
    # Compute Z matrix.
    Z = np.zeros((L, L))
    for i in range(L):
        for j in range(i):  # j < i.
            Z[i, j] = (epsilon[i] + epsilon[j])/(epsilon[i]-epsilon[j])
            Z[j, i] = -Z[i, j]
    Grg = 1./(L-2*N+1)
    if 1-N+L/2 != 0:
        Gp = 1./(1-N+L/2)
    else:
        Gp = np.nan
    if Gisg:
        g_final = G
    else:
        lambd = 1/(1 + G*(N - L/2 - 1))
        g_final = -G*lambd
        if L != 2*N:
            grg = -Grg/(1+Grg*(N-L/2-1))
        else:
            grg = np.nan
        if G < start*Gp < 0: # need to do some trickz
            print('Using inverted g stuff')
            deltas, g_path = use_g_inv(L, N, G, Z, g_step, start=start)
            return g_path, deltas, Z
        elif G > start*Gp > 0: # need to do similar trixkcx
            print('This is not going to work right now. Woops')
            return 'AAAAAAAAA'
    g_path = make_g_path(g_final, g_step)
    G_path = -g_path/(1+g_path*(N-L/2-1))
    # Initial values for Delta with g small. The -2 values (initially
    # occupied states) go where the epsilons are smallest.
    deltas = np.zeros((len(g_path), L), np.float64)
    eps_min = np.argsort(epsilon)
    deltas[0][eps_min[:N]] = -2

    # Finding root while varying g, using prev. solution to start
    for i, g in enumerate(g_path[1:]):
        print('{} out of {} steps'.format(i+1, len(g_path)))
        delta = deltas[i]
        if i > 1:
            corr = (deltas[i] - deltas[i-1])*(g_path[i+1] - g_path[i])/(g_path[i] - g_path[i-1])
            delta = delta + corr
        sol = root(delta_relations, delta, args=(L, N, Z, g, Gamma),
                method='lm', options={'xtol':TOL})
        deltas[i+1] = sol.x
        # checking if we satisfy the delta relations
        dr = delta_relations(deltas[i+1], L, N, Z, g, Gamma)
        errors = np.abs(dr[:-1]/np.mean(np.abs(deltas[i+1])))
        if np.max(errors)> 10**-10:
            print('WARNING: At G= {} error is'.format(G_path[i]))
            print(errors)
    return g_path, deltas, Z


def compute_infinite_G(L, N, epsilon, g_step):
    g = 1./(1-N+L/2)
    g_path, deltas, Z = compute_hyperbolic_deltas(L, N, g, epsilon, g_step, Gisg=True)
    G_path = -g_path/(1+g_path*(N-L/2-1))
    print(G_path)
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
    l = len(g_path)
    # Now forming eigenvalues of IM and observables
    nsk = np.zeros((l, L))
    for i, g in enumerate(g_path):
        nsk[i] = -0.5*deltas[i] + 0.5*g*der_deltas[i]
    return G_path, nsk


def compute_hyperbolic_energy(L, N, G, epsilon, g_step=.01,
                              start=0.9, use_fd=True):
    g_path, deltas, Z = compute_hyperbolic_deltas(L, N, G,
                                                  epsilon, g_step,
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
        # ioms[i][-1] = 2*ioms[i][-1] - ioms[i][-2]
        energies[i] = (1/lambds[i] * np.dot(epsilon, ioms[i])
                + np.sum(epsilon)*(1./2 - 3/4*G_path[i]))
        nsk[i] = -0.5 * deltas[i] + 0.5*g*der_deltas[i]
    return energies, nsk, deltas, G_path, Z


def rgk_spectrum(L, t1, t2, peri=False, fix=False):
    r = np.array([(i+1.0)/L for i in range(L)], dtype=np.float64)
    if peri: # we actually need to handle this more carefully
        k1 = np.append(
                    np.array([2*(i+1)*np.pi/L for i in range(int(L/2-1))],
                    np.float64), 0)
        # k2 = np.append(
                    # np.array([-2*(i+1)*np.pi/L for i in range(int(L/2-1))],
                    # np.float64), -np.pi)
        # k = np.sort(np.concatenate((k2,k1)))
        k = k1
    else: # antiperiodic bc
        k1 = np.array(
                [(2*i+1)*np.pi/L for i in range(int(L/2))],
                np.float64)
        # k2 = np.array(
                # [-(2*i+1)*np.pi/L for i in range(int(L/2))],
                # np.float64)
        # k = np.sort(np.concatenate((k2, k1)))
        k = k1
    eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2)**2))
    epsilon = eta**2
    if fix:
        # l = int(L/20)
        # epsilon[-l:] = epsilon[-l:] + (epsilon[-l:]**2 - epsilon[-l]**2)
        # epsilon[-1] = 2*epsilon[-1]
        epsilon[-1] = 0
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
