import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from solve_rg_model import rgk_spectrum
from solve_rg_model import compute_hyperbolic_energy
from exact_diag import form_ferm_hamiltonian


def Skz(theta):
    sk = 0.5
    skz = -sk*(1-np.abs(theta)**2)/(1+np.abs(theta)**2)
    return skz

def Skp(theta):
    sk = 0.5
    skp = 2*sk*np.conj(theta)/(1+np.abs(theta)**2)
    return skp

def Skm(theta):
    sk = 0.5
    skm = 2*sk*theta/(1+np.abs(theta)**2)
    return skm

def thetatheta(theta):
    return np.prod(1+np.abs(theta)**2)

def Etheta(G, theta, epsilon, norm=False):
    skz = Skz(theta)
    # print(skz)
    skp = Skp(theta)
    skm = Skm(theta)
    term1 = np.sum(epsilon*skz)
    term2 = 0
    L = len(epsilon)
    for i in range(L):
        for j in range(L):
            term2 = term2 + np.sqrt(epsilon[i])*np.sqrt(epsilon[j])*skp[i]*skm[j]
    term3 = np.sum(epsilon*(0.5+skz)**2)
    E = term1 - G*(term2+term3)
    if not norm:
        return E
    else:
        n = thetatheta(theta)
        return E/n


def minimize_me_Etheta(x, L, N, G, epsilon):
    theta_r = x[:L]
    theta_i = x[L:-2]
    theta = theta_r + 1j*theta_i
    E = Etheta(G, theta, epsilon, norm=False)
    norm = thetatheta(theta)
    L = len(epsilon)
    Sz = N-L/2
    # Sz = 0
    return np.abs(E - Sz*x[-2] - norm*x[-1])

def lazy_mf(L, N, Gf, epsilon, steps):
    Gs = Gf*np.linspace(0, 1, steps)
    theta0 = np.zeros(L, np.complex128)
    theta0[:N] = 10
    E0 = np.sum(epsilon[:N])
    Es = np.zeros(steps)
    x = np.concatenate((np.real(theta0), np.imag(theta0)))
    x = np.concatenate((x, [1, E0]))
    for i, G in enumerate(Gs):
        sol = sp.optimize.minimize(minimize_me_Etheta, x, args=(L, N, G, epsilon))
        x = sol.x
        # x = x + 0.1*(0.5 - np.random.rand(2*L+2))
        # sol = sp.optimize.minimize(minimize_me_Etheta, x, args=(L, N, G, epsilon))
        # x = sol.x
        theta = x[:L] + 1j*x[L:-2]
        print(theta)
        E = x[-1]*np.prod(1+np.abs(theta)**2)
        if i == 0: # not starting at the right point woops
            E0 = E0 - E
        # print(np.imag(E))
        # Es[i] = x[-1]
        Es[i] = E + E0
        print('<H> -E = ')
        print(minimize_me_Etheta(x, L, N, G, epsilon))
    return Es, Gs

def form_sd_state(L, N):
    bstate = np.zeros(L)
    bstate[:N] = 1
    fstate = np.concatenate((bstate[::-1], bstate))
    return fstate

def get_norm(N, R):
    norm = np.sum(np.sum(R[:,:]*np.conj(R[:,:]), axis=1))
    # norm2 = np.trace(R*np.conj(R)) # same if R is diagonal but it shouldn't be
    return norm


def hf_energy(R, L, N, G, epsilon):
    term1 = np.sum(epsilon[L:]*np.sum(R[:,:]*np.conj(R[:,:]), axis=1))
    # term1 = np.sum(epsilon[L:]*np.sum((np.abs(R)**2)[:,:N], axis=1))
    # term1 = np.dot(epsilon[L:], np.sum((R*np.conj(R))[:,:N], axis=1))
    term22 = 0
    sqeps = np.outer(np.sqrt(epsilon), np.sqrt(epsilon))
    for i in range(L):
        for j in range(L):
            term22 = term22 - G*sqeps[L+i,L+j]*(np.dot(R[i,:], np.conj(R[j,:]))**2)
            # for a in range(N):
                # for b in range(N):
                    # term2 = term2 - G*sqeps[L+i, L+j]*R[i,a]*R[j,b]*np.conj(R[i,b])*np.conj(R[j,a])
    # print(term22-term2)
    energy = term1 + term22
    return energy


def minimize_me(Rc, L, N, G, epsilon):
    lm = Rc[-1]
    # Rreal = Rc[:L**2].reshape((L, L))
    # Rimag = Rc[L**2:].reshape((L, L))
    # R = Rreal + 1j*Rimag
    R = RtoRm(Rc[:-1], L)
    energy, norm = hf_energy(R, L, N, G, epsilon)
    return energy - norm*lm

def RtoRm(R,L):
    return R[:L**2].reshape((L,L)) + 1j*R[L**2:].reshape((L,L))


def der_f(Rc, L, N, G, epsilon):
    ders = np.zeros((L,L), dtype=np.complex128)
    R = RtoRm(Rc[:-1], L)
    lm = Rc[-1]
    sqeps = np.sqrt(epsilon)
    for j in range(L):
        for s in range(N):
            ders[j,s] = (epsilon[j]-lm)*R[j, s] + 2*G*sqeps[j]*np.sum(
                    sqeps*np.sum(R*np.conj(R[j,:]), axis=1))
    derlm = -np.sum(R[:N]*np.conj(R[:N]))
    df = ders.flatten()
    out = np.append(np.concatenate((np.real(df), np.imag(df))), np.real(derlm))
    return out

def var_eqs(Rc, L, N, G, epsilon):
    eqs = np.zeros((L, L), dtype=np.complex128)
    R = RtoRm(Rc[:-1], L)
    lm = Rc[-1]
    sqeps = np.sqrt(epsilon)
    for j in range(L):
        for s in range(N):
            reqs[j,s] = (epsilon[j] - lm)*R[j,s] + 2*G*sqeps[j]*np.dot(
                    sqeps*(np.conj(R[:,s])*np.sum(R*np.conj(r[j,:]), axis=1)))
    lm_eq = np.sum(R[:,:N]*np.conj(R[:,:N])) - N
    out = np.append(np.concatenate((np.real(df), np.imag(df))), np.real(derlm))
    return out


def good_hf(L, N, Gf, epsilon, steps):
    hepsilon = epsilon[L:]
    diags = np.zeros(L, dtype=np.complex128)
    diags[:N] = 0
    R0 = np.diag(R0)
    Gs = Gf*np.linspace(0, 1, steps)
    R = R0.flatten()
    Rc = np.append(np.concatenate((np.real(R), np.imag(R))), np.sum(hepsilon[:N]))
    # Rs = np.zeros((len(Rc), steps))
    Rs = np.zeros((steps, len(Rc)))

    hf_energies = np.zeros(steps)
    pw_energies = np.zeros(steps)
    for i, G in enumerate(Gs):
        print('G:')
        print(G)
        sol = sp.optimize.root(var_eqs, Rc, args=(L, N, G, hepsilon))
        Rc = sol.x
        Rs[i] = Rc
        Rm = RtoRm(Rc[:-1], L)
        hf_energies[i], norm = hf_energy(Rm, L, N, G, epsilon)
        print('HF energy: {}'.format(hf_energies[i]))
        pw_energies[i], _ = hf_energy(R0, L, N, G, epsilon)
        print('PW energy: {}'.format(pw_energies[i]))
    return Gs, hf_energies, pw_energies, Rm




def lazy_hf(L, N, Gf, epsilon, steps):
    Rdiag = np.zeros(L, dtype=np.complex128)
    Rdiag[:N] = 1
    R0 = np.diag(Rdiag)

    Gs = Gf*np.linspace(0, 1, steps)
    energies = np.zeros(steps)
    energies_pw = np.zeros(steps)
    E0 = np.sum(epsilon[L:L+N])
    R = R0.flatten()
    R = np.concatenate((np.real(R), np.imag(R)))
    Rc = np.append(R, E0)
    lms = np.zeros(steps)
    print('E0 is {}'.format(E0))
    for i, G in enumerate(Gs):
        print('G:')
        print(G)
        # cons = ({'type': 'eq',
                 # 'fun': lambda x: (np.dot(RtoRm(x, L), np.conj(RtoRm(x, L).T)) - np.diag(np.ones(L))).flatten()
                 # })
        sol = sp.optimize.minimize(minimize_me, Rc, args=(
                                   L, N, G, epsilon))
        Rc = sol.x
        # now repeating after adding some noise
        # Rc = Rc + 0.01*(0.5-np.random.rand(2*L**2))
        # cons = ({'type': 'eq',
                 # 'fun': lambda x: (np.dot(RtoRm(x, L), np.conj(RtoRm(x, L).T)) - np.diag(np.ones(L))).flatten()
                 # })
        # sol = sp.optimize.minimize(minimize_me, Rc, args=(
                                   # L, N, G, epsilon), constraints=[cons])
        # Rc = sol.x
        R = RtoRm(Rc[:-1], L)
        e, norm = hf_energy(R, L, N, G, epsilon)
        e = e
        print('norm: {}'.format(norm))
        print('E: {}'.format(e))
        energies[i] = e
        energies_pw[i], _ = hf_energy(R0, L, N, G, epsilon)
        print('Epw: {}'.format(energies_pw[i]))
        # lms[i] = Rc[-1]*norm
    return Gs, energies, energies_pw, lms, R


def n_k(R, L, N):
    return np.sum((R*np.conj(R))[:,:N], axis=1)


if __name__ == '__main__':
    L = 4
    N = 3
    G = 1.5/(L-2*N+1)
    k, hepsilon = rgk_spectrum(2*L, 1, 0)
    Eexact, n, deltas, Gexact, Z = compute_hyperbolic_energy(
            L, N, G, hepsilon, .1/L)
    epsilon = np.concatenate((hepsilon[::-1], hepsilon))
    steps = 20
    # energies, Gs = lazy_mf(L, N, G, hepsilon, steps)
    Gs, energies, energies_pw, R = good_hf(L, N, G, epsilon, steps)

    energies2 = (1-Gs)*np.sum(hepsilon[:N])
    plt.scatter(Gs, energies)
    plt.scatter(Gs, energies2)
    plt.scatter(Gs, energies_pw, marker='x')
    # plt.scatter(Gs, lms, marker='x')
    plt.scatter(Gexact, Eexact)
    plt.show()

    plt.subplot(2,1,1)
    sns.heatmap(np.real(R))
    plt.subplot(2,1,2)
    sns.heatmap(np.imag(R))
    plt.show()

    # nk = n_k(R, L, N)
    # plt.scatter(k, nk)
    # plt.show()
