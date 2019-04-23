import numpy as np
from scipy.special import factorial

def make_X_Z(L, epsilon, r=None):
    X = np.zeros((L,L))
    Z = np.zeros((L, L))
    if r is None:
        r = 2*epsilon[-1]
    for i in range(L):
        for j in range(i):
            if i != j: # else, already zero
                X[i, j] = 2*np.sqrt(epsilon[i]*epsilon[j])/(
                           epsilon[i] - epsilon[j])
                Z[i, j] = (epsilon[i] + epsilon[j])/(
                           epsilon[i] - epsilon[j])
                X[j, i] = - X[i, j]
                Z[j, i] = - Z[j, i]
    Xr = 2*np.sqrt(epsilon)*np.sqrt(r)/(epsilon - r)
    Zr = (epsilon + r)/(epsilon - r)
    return X, Z, Xr, Zr


def int_eqn(Ik, alphak):
    return np.sum(Ik) - alphak


def slater_det_ip(sd_inds, Lambda, epsilon, L, r=-1):
    N = len(sd_inds)
    X, Z, Xr, Zr = make_X_Z(L, epsilon)
    J = np.zeros((L, L))
    diags = [Lambda[i] - np.sum([Z[i,j] for j in sd_inds]) + Zr[i] for i in sd_inds]
    J = np.diag(diags) + X[sd_inds, sd_inds]
    XProd = np.prod([Xr[i] for i in sd_inds])
    return np.linalg.det(J)/XProd


def S_Plus(Lambda, k, epsilon, g):
    """ computes <e_alpha' |S_k^+ | e_alpha> """
    """ arbitrary r is 0 """
    L = len(epsilon)
    J = np.zeros((L,L))
    X, Z, Xr, Zr = make_X_Z(L, epsilon)
    for i in range(L):
        for j in range(L):
            if i == j and i != k:
                J[i, j] = 2*Lambda[i] + 2/g - np.sum(Z[i, :]) + Z[i, k] + Zr[i]
            elif i != k and j != k:
                J[i, j] = X[i, j]
    Xprod = np.prod(Xr[i])/Xr[k]
    return np.linalg.det(J)/Xprod


def get_norm(Lambda, epsilon, L, N, g):
    X, Z, Xr, Zr = make_X_Z(L, epsilon)
    LambdaDual = Lambda + 2/g
    diags = 2*Lambda + 2/g - np.sum(Z, axis=1) + Zr
    J = np.diag(diags) + X
    XProd = np.prod(Xr)
    NormProd = np.linalg.det(J)/XProd
    ref_inds = [i for i in range(N)]
    Overlap = slater_det_ip(ref_inds, Lambda, epsilon, L)
    OverlapDual = slater_det_ip(ref_inds, LambdaDual, epsilon, L)

    norm2 = NormProd*Overlap/OverlapDual

    return np.sqrt(np.abs(norm2))


if __name__ == '__main__':
    from solve_rg_model import compute_hyperbolic_deltas, rgk_spectrum
    import sys
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    Gr = float(sys.argv[3])
    ks, epsilon = rgk_spectrum(2*L, 1, 0)
    G = Gr/(L-2*N+1)
    g_path, deltas, Z = compute_hyperbolic_deltas(L, N, G, epsilon, .1/L)
    g = g_path[-1]
    Lambda = deltas[-1]/g

    Norm = get_norm(Lambda, epsilon, L, N, g)
    print(Norm)

    sd_inds = [i for i in range(N)]
    ip = slater_det_ip(sd_inds, Lambda, epsilon, L)
    print(ip/(Norm*np.sqrt(factorial(N))))

    sd_inds = [i for i in range(N)]
    sd_inds[-1] = sd_inds[-1] + 1
    ip = slater_det_ip(sd_inds, Lambda, epsilon, L)
    print(ip/(Norm*np.sqrt(factorial(N))))

    # SPlus = S_Plus(Lambda, L-3, epsilon, g)
    # print(SPlus/Norm)
