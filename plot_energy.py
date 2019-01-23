import numpy as np
import matplotlib.pyplot as plt
from solve_rg_model import compute_iom_energy_quad, compute_iom_energy
from solve_rg_model import compute_hyperbolic_energy
from exact_diag import compute_n_exact, compute_E


def Gofg(g, L, N, A, B, C, seps):
    M = N - L/2
    lamb = 1+B*g*(M-1)
    G = -g/(2*lamb - g*C*seps)
    return G


def find_Gc(A, B, C, L, N, seps):
    N = L//4
    gamma = np.sqrt(B**2-A*C)
    gc = -1./(gamma*(L/2-1)) # crit point from my work
    gmr = -1./(gamma*(L/2-N-1))
    Gc = Gofg(gc, L, N, A, B, C, seps)
    Gmr = Gofg(gmr, L, N, A, B, C, seps)
    return Gc, Gmr


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
    print('Critical g is {}'.format(Gc*L))
    gcc = -Gc/(1-Gc+Gc*(N-L/2))
    print('In the numerics this is g = {}'.format(gcc))
    Gmr = 1./(L-N+1)
    print('Moore-Read line is at {}'.format(Gmr*L))
    Gp = 1./(1-N+L/2)
    print('We will have trouble around g={}'.format(
        Gp*L))
    print('New weird point is g= {}'.format(2.0*L/(N-1)))
    Gnew = 2.0/(N-1)
    if Gc > 0:
        Gs = np.linspace(-1.5*Gc, 1.5*Gc, steps)
    else:
        Gs = np.linspace(5*Gc, 0.5*Gc, steps)
    # Gs = np.linspace(5*Gc, 0.75*Gc, steps)
    gs = L*Gs
    energy1 = np.zeros(steps)
    energy2 = np.zeros(steps)
    energy3 = np.zeros(steps)
    for i, G in enumerate(Gs):
        print('{}th step, G = {}'.format(i, G))
        # if G !=0:
            # gsteps = int(np.abs(G)/g_step)
        # else:
            # gsteps = 1
        # # print(gsteps)
        # energy1[i] = compute_iom_energy(L, N,
                # G, 'hyperbolic', epsilon, steps = gsteps,
                # return_n = False,
                # taylor_expand = False)
        energy1[i], n, d1, success = compute_hyperbolic_energy(
                L, N, G, epsilon, g_step=g_step,
                taylor_expand=False, holdover=0)
        # energy2[i], n, d2, success = compute_hyperbolic_energy(
                # L, N, G, epsilon, g_step=g_step,
                # taylor_expand=False, holdover=0.1)
        # energy3[i], n, d3, success = compute_hyperbolic_energy(
                # L, N, G, epsilon, g_step=g_step,
                # taylor_expand=False, holdover=0.25)

    denergy = np.gradient(energy1, gs)
    d2energy = np.gradient(denergy, gs)
    d3energy = np.gradient(d2energy, gs)
    if L < 13: # small enough to diagonalize
        eenergy = np.zeros(steps)
        for i in range(steps):
            ee, _ = compute_n_exact(L, N, Gs[i], epsilon)
            eenergy[i] = ee
        dee = np.gradient(eenergy, gs)
        d2ee = np.gradient(dee, gs)
        d3ee = np.gradient(d2ee, gs)

    plt.subplot(3,1,1)
    plt.plot(gs, energy1/L)
    plt.scatter(gs, energy1/L, s=5)
    # plt.axvline(-L*Gnew)
    # plt.plot(gs, energy2/L, label='Holdover .25', linestyle = ':')
    # plt.plot(gs, energy3/L, label='Holdover .5', linestyle = ':')
    if L <13:
        plt.plot(gs, eenergy/L, linestyle = ':',
                color = 'c',
                label='Diagonalization')
        plt.legend()
    # plt.xlabel('G*L')
    plt.xlabel('g')
    plt.title('L = {}, N = {}'.format(L, N))

    plt.subplot(3,1,2)
    plt.plot(gs[5:-5], d3energy[5:-5]/L)
    plt.scatter(gs[5:-5], d3energy[5:-5]/L)
    # plt.axvline(-L*Gnew)
    # if L < 13:
        # plt.plot(gs[:l], d3ee/L,
                # linestyle=':',
                # color = 'c',
                # label='Diagonalization')
    plt.axvline(Gc*L, color = 'r')
    # plt.axvline(Gmr*L, color = 'b')
    plt.xlabel('g')
    plt.ylabel('d3e/dg')
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
