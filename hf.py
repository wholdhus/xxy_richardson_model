import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from celluloid import Camera
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


def hf_energy(Rc, L, N, G, epsilon):
    R = np.sin(np.real(Rc))*np.exp(1j*np.imag(Rc))
    term1 = 0.5*np.dot(epsilon[L:], np.sum(R[L:,L:L+N]*np.conj(R[L:,L:L+N]), axis=1))
    term1 = term1 + 0.5*np.dot(epsilon[:L], np.sum(R[:L,L-N:L]*np.conj(R[:L,L-N:L]), 
                    axis=1))
    # term1 = np.sum(epsilon[L:]*np.sum((np.abs(R)**2)[:,:N], axis=1))
    # term1 = np.dot(epsilon[L:], np.sum((R*np.conj(R))[:,:N], axis=1))
    term22 = 0
    sqeps = np.outer(np.sqrt(epsilon), np.sqrt(epsilon))
    for i in range(L):
        for j in range(L):
            term22 = term22 - G*sqeps[L+i,L+j]*(np.dot(R[L+i,L:L+N], np.conj(R[L+j,L:L+N]))
                    *np.dot(R[L-1-i,L-1-N:L], np.conj(R[L-1-j,L-1-N:L])))
    energy = term1 + term22
    norm=0.5*np.sum(np.sum(R[L:, L:]*np.conj(R[L:, L:]))
            +np.sum(R[:L,:L]*np.conj(R[:L,:L])))/L
    return np.real(energy), np.real(norm)


def minimize_me(R1d, L, N, G, epsilon):
    # R = R1d[:4*L**2].reshape(2*L, 2*L) + 1j*R1d[4*L**2:-1].reshape(2*L, 2*L)
    R = R1d[:4*L**2].reshape(2*L, 2*L)
    energy, norm = hf_energy(R, L, N, G, epsilon)
    return np.abs(energy - R1d[-1]*norm)


def lazy_hf(L, N, Gf, steps):
    k, hepsilon = rgk_spectrum(2*L, 1, 0)
    epsilon = np.concatenate((hepsilon[::-1], hepsilon))
    print(np.shape(epsilon))
    # diags0 = np.zeros(L)
    # diags0[:N] = 1
    # R0 = np.diag(np.concatenate((diags0[::-1], diags0)))
    # R0 = np.zeros((2*L, 2*L))
    # diags = np.zeros(L)
    # diags[:N] = np.pi/2
    diags = np.ones(L)*np.pi/2
    R0 = np.diag(np.concatenate((diags[::-1], diags)))

    Gs = np.linspace(0, 1, steps)*Gf
    energies = np.zeros(steps)
    energies_pw = np.zeros(steps)
    energies_lm = np.zeros(steps)

    Ee, n, delta, Ge, Z = compute_hyperbolic_energy(L, N, Gf, hepsilon)

    E0 = np.sum(epsilon[L:L+N])
    # R1d = np.append(
            # np.concatenate(
                # (R0.flatten(), np.zeros(4*L**2))),
            # E0)
    R1d = np.append(R0.flatten(), E0)
    print('Initial guess energy:')
    E00, norm = hf_energy(R0, L, N, 0, epsilon)
    print(E00)
    print(E0)
    print(norm)
    for i, G in enumerate(Gs):
        # if i < 10:
        noiseM = 0.05*(0.5-np.random.rand(2*L, 2*L))
        # noiseM = np.diag(np.diag(noiseM, 1), 1)
        # noiseM = noiseM + 0.05*np.diag(np.diag(noiseM, 2), 2)
        noiseM = np.triu(noiseM, 1)
        noiseM = noiseM - noiseM[::-1, ::-1] # has shape that looks like what we want
        # print(np.round(np.sin(noiseM), 2))
        # noise = np.concatenate((noiseM.flatten(), np.ones(4*L**2)*np.pi/2))
        # noise = np.append(noise, 0)
        noise = np.append(noiseM.flatten(), 0)
        last = R1d
        R1d = R1d + noise
        sol = sp.optimize.minimize(minimize_me, R1d, args=(L, N, G, epsilon))
        if i > 0:
            R1d = 0.9*R1d + 0.1*last
        # R1d = sol.x + (sol.x-R1d)*steps/Gf
        # R = R1d[:4*L**2].reshape(2*L, 2*L) + 1j*R1d[4*L**2:-1].reshape(2*L, 2*L)
        R = R1d[:4*L**2].reshape(2*L, 2*L)

        E, norm = hf_energy(R, L, N, G, epsilon)
        E = E/norm
        energies[i] = E
        energies_pw[i], _ = hf_energy(R0, L, N, G, epsilon)
        energies_lm[i] = R1d[-1] * norm
        print('')
        print('G={}'.format(G))
        print('Norm = {}'.format(norm))
        print('Ehf = {}'.format(E))
    plt.figure(figsize=(12,8))
    plt.scatter(Gs, energies, color = 'c')
    plt.scatter(Gs, energies_pw, color = 'm')
    plt.scatter(Gs, energies_lm, color ='c', marker='x', s=40)
    energies0 = (1-Gs)*np.sum(epsilon[L:L+N])
    plt.scatter(Gs, energies0, color = 'm', marker='x', s=40)

    plt.scatter(Ge, Ee)
    plt.show()

    plt.subplot(2,1,1)
    sns.heatmap(np.sin(np.real(R)))
    plt.subplot(2,1,2)
    sns.heatmap(np.imag(R))
    plt.show()


def RtoRm(R,L):
    return R[:L**2].reshape((L,L)) + 1j*R[L**2:].reshape((L,L))

def var_eqs(thetasC, L, N, G, epsilon):
    eqs = np.zeros((L, L))
    theta = thetasC[:-1].reshape((L,L))
    lm = thetasC[-1]
    sqeps = np.sqrt(epsilon)
    lm_eq = -N
    for j in range(L):
        for s in range(L):
            eqs[j,s] = ((epsilon[j] - lm)*np.sin(2*theta[j,s])
                    - 4*G*sqeps[j]*np.cos(theta[j,s])*np.dot(
                        sqeps*np.sin(theta[:,s]),
                            np.sum(np.sin(theta[j,:])*np.sin(theta), axis=1)))
            lm_eq = lm_eq + (np.sin(theta[j,s]))**2
    # out = np.append(np.concatenate((np.real(eqs), np.imag(eqs))), np.real(lm_eq))
    out = np.append(eqs, lm_eq)
    return out


def good_hf(L, N, Gf, epsilon, steps):
    hepsilon = epsilon[L:]
    diags = np.zeros(L)
    diags[:N] = np.pi/2
    theta0 = np.diag(diags)
    Gs = Gf*np.linspace(0, 1, steps)
    thetaC = theta0.flatten()
    lm = (epsilon[N]-epsilon[N-1])/2
    thetaC = np.append(thetaC, lm)
    # Rs = np.zeros((len(Rc), steps))
    # thetas = np.zeros((steps, len(Rc)))


    hf_energies = np.zeros(steps)
    pw_energies = np.zeros(steps)
    for i, G in enumerate(Gs):
        print('G:')
        print(G)
        rands = np.append((1-0.5*np.random.rand(L**2)), 0)
        thetaC = thetaC + 0.5*rands
        sol = sp.optimize.root(var_eqs, thetaC, args=(L, N, G, hepsilon))
        thetaC = 0.1*sol.x + 0.9*thetaC
        theta = thetaC[:-1].reshape((L,L))
        hf_energies[i] = hf_energy(np.sin(theta), L, N, G, epsilon)
        print('HF energy: {}'.format(hf_energies[i]))
        pw_energies[i] = hf_energy(np.sin(theta0), L, N, G, epsilon)
        print('PW energy: {}'.format(pw_energies[i]))
        print('HF lm: {}'.format(thetaC[-1]))
    return Gs, hf_energies, pw_energies, theta


def bcs_delta_eq(DeltaR, L, N, G, epsilon):
    Delta = DeltaR[:L] + 1j*DeltaR[L:]
    ef = (epsilon[N] + epsilon[N-1])/2
    # ef = epsilon[N-1]
    # ef = np.sum(epsilon[:N])
    eqs = 0.5*G*np.sqrt(epsilon)*np.sum(np.sqrt(epsilon)*Delta/np.sqrt(Delta**2+(epsilon-ef)**2)) - Delta
    Ek = np.sqrt(np.abs(Delta**2) + (epsilon-ef)**2)
    # n_eq = np.sum(0.5*(1-(epsilon-ef)/Ek))-N
    # return np.append(np.concatenate((np.real(eqs), np.imag(eqs))), n_eq)
    return np.concatenate((np.real(eqs), np.imag(eqs)))


def do_bcs(L, N, Gf, steps):
    fig = plt.figure(figsize=(12,8))
    camera = Camera(fig)
    k, epsilon = rgk_spectrum(2*L, 1, 0)
    ef = (epsilon[N] + epsilon[N-1])/2
    # ef = epsilon[N-1]
    # ef = np.sum(epsilon[:N])
    Delta = np.zeros(2*L, np.float64)
    Delta[:N] = (epsilon[:N]-ef)
    Gs = Gf*np.linspace(0, 1, steps)
    Deltas = np.zeros((steps, 2*L))
    vks = np.zeros((steps, L))
    uks = np.zeros((steps, L))
    energies = np.zeros(steps)
    for i, G in enumerate(Gs):
        sol = sp.optimize.root(bcs_delta_eq, Delta, args=(L, N, G, epsilon),
                               method='lm')
        tries = 0
        Delta = sol.x
        while np.abs(Delta[N]) < 0.1/L  and tries < 10 and i > 0:
            print('retrying {}'.format(tries))
            noise = (tries+1)*100*L*(0.5 - np.random.rand(2*L))
            Delta = Delta + noise
            sol = sp.optimize.root(bcs_delta_eq, Delta, args=(L, N, G, epsilon),
                                   method='lm')
            Delta = sol.x
            tries = tries + 1
        Deltas[i] = Delta
        cDelta = Delta[:L] + 1j*Delta[L:]
        Ek = np.sqrt(np.abs(cDelta)**2 + (epsilon-ef)**2)
        vks[i] = 0.5*(1-(epsilon-ef)/Ek)
        uks[i] = 0.5*(1+(epsilon-ef)/Ek)
        if i%5 == 0 and i > 20:
            plt.scatter(k, vks[i], color = 'c')
            plt.scatter(k, uks[i], color = 'm')
            # plt.scatter(k, np.real(cDelta), color = 'c')
            # plt.scatter(k, np.imag(cDelta), color = 'm')
            # camera.snap()
        ukvk = cDelta/(2*Ek)
        E = 2*np.dot(epsilon-ef, vks[i]) - G*np.sum(np.sqrt(epsilon)*ukvk)**2
        energies[i] = E
        print('N:')
        print(np.sum(vks[i]))
    # animation = camera.animate()
    # animation.save('animation.mp4')
    plt.show()
    plt.scatter(Gs, energies)
    plt.show()


if __name__ == '__main__':
    L = 4
    N = 3
    G = 1.5/(L-2*N+1)
    steps = 100
    lazy_hf(L, N, G, steps)
