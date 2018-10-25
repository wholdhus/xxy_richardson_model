import numpy as np
import matplotlib.pyplot as plt
from solve_rg_model import compute_iom_energy_quad, compute_iom_energy
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
                    steps=50, diag=True, filename=None,
                    escale=None, gsteps=100,
                    use_fixed=True):
    model = 'hyperbolic'
    # k = np.pi*np.linspace(0.0, L, L)/L
    # epsilon = k**2
    t1 = 1
    t2 = 0
    k = np.pi*np.linspace(0.0, L, L)/L
    eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2))**2)
    epsilon = eta**2

    if escale is not None:
        epsilon = epsilon*escale
    N = int(L*dens)
    Gc = 1./(L - 2*N + 1)
    print('Critical g is {}'.format(Gc*L))
    gcc = -Gc/(1-Gc+Gc*(N-L/2))
    print('In the numerics this is g = {}'.format(gcc))
    Gp = 1./(1-N+L/2)
    print('We will have trouble around g={}'.format(
        Gp*L))
    if Gc > 0:
        Gs = np.linspace(-1.5*Gc, 1.5*Gc, steps)
    else:
        Gs = -np.linspace(1.5*Gc, -1.5*Gc, steps)
    gs = L*Gs
    energy = np.zeros(steps)
    for i in range(steps):
        if L < 13:
            te = True
        else:
            te = False # there are problems with t.e. for large systems
        eq = compute_iom_energy(L, N, Gs[i], model, epsilon, steps=gsteps,
                taylor_expand=False, return_n=False,
                use_fixed_rels=use_fixed)
        energy[i] = eq

    denergy = np.gradient(energy, gs)
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

    plt.subplot(2,1,1)
    plt.plot(gs, energy/L, label='energy', color = 'black')
    if L <13:
        plt.plot(gs, eenergy/L, linestyle = ':',
                color = 'c',
                label='energy from diag')
        plt.legend()
    plt.xlabel('G*L')
    plt.ylabel('E/L')
    plt.title('L = {}'.format(L))

    plt.subplot(2,1,2)
    l = len(d3energy)
    plt.plot(gs[:l], d3energy/L,
            label='3rd deriv from quads', color = 'black')
    if L < 13:
        plt.plot(gs[:l], d3ee/L,
                linestyle=':',
                color = 'c',
                label='3rd deriv from diag')
        plt.legend()
    plt.axvline(Gc*L)
    plt.xlabel('G*L')
    plt.ylabel('E\'\'\'/L')


    if filename is not None:
        import pandas as pd
        df = pd.DataFrame(
                {'g': gs, 'E_quad': energy, 'd3E_quad': d3energy})
        df.to_csv(filename)


def plot_energy_derivs(A, B, C, L, steps=11):
    gmax = 1
    N = L//2
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    # epsilon = -2*np.cos(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    energy = np.array([0. for i in range(steps)])
    denergy = np.array([0. for i in range(steps)])
    # G = np.linspace(-gmax/L, gmax/L, steps)
    G = np.linspace(0.00001, gmax/L, steps)
    for i in range(steps):
        energy[i], denergy[i]= compute_iom_energy_quad(
            L, N, G[i], A, B, C, epsilon)
    denergyalt = np.gradient(energy, G)
    d2energy = np.gradient(denergy, G)
    d3energy = np.gradient(d2energy, G)

    plt.subplot(4, 1, 1)
    plt.plot(G*L, energy/L, label = "a = {0}, b = {1}, c={2}, L={3}".format(
        A, B, C, L))
    plt.subplot(4, 1, 2)
    plt.plot(G*L, denergy/L)
    plt.plot(G*L, denergyalt/L)

    plt.subplot(4, 1, 3)
    plt.plot(G*L, d2energy/L)

    plt.subplot(4, 1, 4)
    plt.plot(G*L, d3energy/L)
    Gc, Gmr = find_Gc(A, B, C, L, N, np.sum(epsilon))
    plt.axvline(Gc*L)
    plt.axvline(Gmr*L)


if __name__ == '__main__':
    L = int(input('Number of sites: '))
    dens = float(input('Density: '))
    steps = int(input('Number of steps to increment G: '))
    use_fixed = input('Use fixed delta relations? ')
    if use_fixed == 'Y':
        use_fixed = True
    else:
        use_fixed = False
    filename = None
    sav = input('Save file? Y/N ')
    if sav == 'Y':
        save = True
        filename = input('Savefile: ')
    plot_hyp_energy(L=L, dens=dens, steps=100,
                    gsteps=steps,
                    filename=filename,
                    use_fixed=use_fixed)
    plt.show()
