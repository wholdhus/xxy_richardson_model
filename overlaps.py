from solve_rg_model import rgk_spectrum, compute_hyperbolic_deltas
import numpy as np

def Z(a,b):
    return (a+b)/(a-b)

def X(a,b):
    return 2*np.sqrt(a*b)/(a-b)

def sd_overlap(L, N, epsilon, lambdas, g):
    J = np.zeros((N, N))
    r = epsilon[0]*0.5
    eps = epsilon[:N]
    Xav = np.mean(np.abs(X(r, eps)))
    Xprod = np.prod(X(r, eps)/Xav)
    for i in range(N):
        Z1 = np.array([Z(eps[i], e) for e in eps[np.arange(N) != i]])
        J[i, i] = Xav*(lambdas[i] - np.sum(Z1) + Z(r, epsilon[i]))
        for j in range(i):
            J[i, j] = X(epsilon[i], epsilon[j])
            J[j, i] = -1*J[i, j]
    return np.linalg.det(J)/Xprod

def find_normalization(L, N, epsilon, lambdas, g):
    olap1 = sd_overlap(L, N, epsilon, lambdas, g)
    olap2 = sd_overlap(L, N, epsilon, lambdas + 2./g, g)
    r = epsilon[0]*0.5

    J = np.zeros((L, L))
    Xav = np.mean(X(r, epsilon))
    for i in range(L):
        Z1 = np.array([Z(epsilon[i], e) for e in epsilon[np.arange(L) != i]])
        J[i, i] = (2*lambdas[i] + 2./g - np.sum(Z1) + Z(r, epsilon[i]))*Xav
        for j in range(i):
            J[i, j] = X(epsilon[i], epsilon[j])
            J[j, i] = -1*J[i, j]
    Xprod = np.prod(X(r, epsilon)/Xav)
    prod = np.linalg.det(J)/Xprod

    N2 = prod * olap1 / olap2
    return np.sqrt(N2)


if __name__ == '__main__':
    L = 12
    N = 4
    g1 = -0.5
    g2 = -0.1
    k, epsilon = rgk_spectrum(L*2, 1, 0)

    G = g1/L
    print(G)
    g_path, deltas, Zs = compute_hyperbolic_deltas(L, N, G, epsilon, .01,
                                                   start=0.9)
    g = G/(1 + G*(N - L/2 - 1))
    lambdas = deltas[-1, :]*g
    soverlap = sd_overlap(L, N, epsilon, lambdas, g)
    print('G=0.5 overlap with sd is ')
    print(soverlap)
    print('GS normalization:')
    norm = find_normalization(L, N, epsilon, lambdas, g)
    print(norm)
    print('Normalized overlap:')
    print(soverlap/(norm))

    G = g2/L
    print(G)
    g_path, deltas, Zs = compute_hyperbolic_deltas(L, N, G, epsilon, .01,
                                                   start=0.9)
    g = G/(1 + G*(N - L/2 - 1))
    lambdas = deltas[-1, :]*g
    soverlap = sd_overlap(L, N, epsilon, lambdas, g)
    print('G=0.1 overlap with sd is ')
    print(soverlap)
    norm = find_normalization(L, N, epsilon, lambdas, g)
    print('GS normalization: ')
    print(norm)
    print('Normalized overlap:')
    print(soverlap/(norm))
