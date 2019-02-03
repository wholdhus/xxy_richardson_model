import numpy as np
import matplotlib.pyplot as plt
from solve_rg_model import compute_iom_energy, compute_hyperbolic_energy
from exact_diag import compute_n_exact, compute_E


def plot_hyp_energy(epsilon=None, L=10, dens = 3/4,
        steps=50, g_step=0.001, diag=True, filename=None):
    # gsteps is the number of steps I will increment
    # G in the numerics. It seems like this should
    # We generally need a higher number of steps for higher
    # coupling, so this should let us do the easy cases quickly.
    t1 = 1
    t2 = 0
    k = np.pi*np.linspace(0, 1, L)
    # eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2))**2)
    # epsilon = eta**2
    # epsilon = -0.5 * t1 * (np.cos(k) - 1) - 0.5 * t2 * (np.cos(2*k) -1)
    epsilon = k**2

    N = int(L*dens)
    Gc = 1./(L - 2*N + 1)
    print('Read-Green line is {}'.format(Gc))
    Gmr = 1./(L-N+1)
    print('Moore-Read line is {}'.format(Gmr))
    Gp = 1./(1-N+L/2)
    print('We will have trouble around G={}'.format(Gp))
    print('New weird point is G= {}'.format(2.0/(N-1)))
    Gnew = 2.0/(N-1)
    if Gp > 0:
        Gs = np.linspace(0, 1.51*Gp, steps)
    else:
        Gs = np.linspace(1.51*Gp, 0, steps)
    gs = L*Gs
    energy1 = np.zeros(steps)
    energy2 = np.zeros(steps)
    for i, G in enumerate(Gs):
        print('')
        print('{}th step, G = {}'.format(i, G))
        energy1[i], n, d1, success = compute_hyperbolic_energy(
                L, N, G, epsilon, g_step=g_step,
                holdover=0, try_g_inv=False)
        energy2[i], n, d2, success = compute_hyperbolic_energy(
                L, N, G, epsilon, g_step=g_step,
                holdover=0, try_g_inv=True)

    denergy = np.gradient(energy1, gs)
    d2energy = np.gradient(denergy, gs)
    d3energy = np.gradient(d2energy, gs)
    if L < 13: # small enough to diagonalize
        e0 = np.zeros(steps)
        e1 = np.zeros(steps)
        e2 = np.zeros(steps)
        e3 = np.zeros(steps)
        for i, G in enumerate(Gs):
            spectrum = compute_E(L, N, G, epsilon)
            spectrum.sort()
            e0[i] = spectrum[0]
            e1[i] = spectrum[1]
            e2[i] = spectrum[2]
            e3[i] = spectrum[3]
        # dee = np.gradient(eenergy, gs)
        # d2ee = np.gradient(dee, gs)
        # d3ee = np.gradient(d2ee, gs)

    # plt.subplot(2,1,1)
    plt.scatter(gs, energy1/L, s=20, label='No holdover')
    plt.scatter(gs, energy2/L, label='No holdover, used g^-1', s=10)
    plt.axvline(Gp*L, color='m')
    plt.axvline(Gc*L, color='g')
    if L <13:
        plt.plot(gs, e0/L, linestyle = ':')
        plt.plot(gs, e1/L, linestyle = ':')
        plt.plot(gs, e2/L, linestyle = ':')
        plt.plot(gs, e3/L, linestyle = ':')
    plt.legend()
    plt.xlabel('g')
    plt.title('L = {}, N = {}'.format(L, N))

    # plt.subplot(2,1,2)
    # plt.plot(gs, d3energy/L)
    # plt.scatter(gs, d3energy/L)
    # plt.axvline(-L*Gnew)
    # if L < 13:
        # plt.plot(gs[:l], d3ee/L,
                # linestyle=':',
                # color = 'c',
                # label='Diagonalization')
    # plt.axvline(Gc*L, color = 'r')
    # plt.axvline(Gmr*L, color = 'b')
    # plt.xlabel('g')
    # plt.ylabel('d3e/dg')
    return Gs, energy1, d3energy

if __name__ == '__main__':
    plt.figure(figsize=(15,8))
    L = int(input('Number of sites: '))
    dens = float(input('Density: '))
    gs = float(input('Step size: '))
    samples = int(input('Number of samples for derivative: '))
    filename = None

    G, E, d3E = plot_hyp_energy(L=L, dens=dens, steps=samples,
                                g_step = gs)
    sav = input('Save file? Y/N ')
    if sav == 'Y':
        save = True
        filename = input('Savefile: ')
        import pandas as pd
        df = pd.DataFrame({'G': G, 'Energy': E, 'd3e/dg3': d3E})
        df.to_csv(filename)
    plt.show()
