import numpy as np
import matplotlib.pyplot as plt
from solve_rg_model import compute_iom_energy_quad, compute_iom_energy

def plot_energies(A, B, C, steps=20):
    Ls = [10, 20, 50, 100]
    gmax= 1
    colors = ['m', 'c', 'r', 'g']
    for j in range(4):
        L = Ls[j]
        print('Trying system size {}'.format(L))
        N = L//2
        eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
        epsilon = eta**2

        energy = np.array([0 for i in range(steps)])
        dEnergy = np.array([0 for i in range(steps)])
        G = np.linspace(0.00001, gmax/L, steps)
        for i in range(steps):
            energy[i], dEnergy[i] = compute_iom_energy_quad(
                    L, N, G[i], A, B, C, epsilon)
        plt.plot(
            G*L, energy/L, label = "L = {}".format(L), color = colors[j])


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
    

def plot_energy_boring_derivs(L, steps=11):
    N = L//4
    Gmr = 1./(L-N+1)
    Gc = 1./(L+1)
    Gc2 = 2
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    print('Moore-read G is {}'.format(Gmr))
    print('Critical G is {}'.format(Gc))
    energy = np.array([0. for i in range(steps)])
    G = np.linspace(0.00001, 3, num=steps)
    for i in range(steps):
        energy[i] = compute_iom_energy(L, N, G[i], 'hyperbolic',
                epsilon, G_step=0.002)
    denergy = np.gradient(energy, G)
    d2energy = np.gradient(denergy, G)
    d3energy = np.gradient(d2energy, G)
    d4energy = np.gradient(d3energy, G)

    plt.subplot(5, 1, 1)
    plt.plot(G, energy/L, label = "Hyperbolic model, L={}".format(L))

    plt.subplot(5, 1, 2)
    plt.plot(G, denergy/L)

    plt.subplot(5, 1, 3)
    plt.plot(G, d2energy/L)

    plt.subplot(5, 1, 4)
    plt.plot(G, d3energy/L)
    
    plt.subplot(5, 1, 5)
    plt.plot(G, d4energy/L)

if __name__ == '__main__':
    # shoudl recreate stuff from stouten??
    plot_energy_derivs(4., 2., 1., 100, steps=100)
    # plot_energy_boring_derivs(20, 100)
    plt.show()
    
