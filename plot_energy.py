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

def plot_energy(A, B, C, L, steps=11):
    gmax = 0.25
    N = L//2
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    energy = np.array([0 for i in range(steps)])
    # G = np.linspace(-gmax/L, gmax/L, steps)
    G = np.linspace(0.00001, gmax/L, steps)
    for i in range(steps):
        energy[i]= compute_iom_energy_quad(
            L, N, G[i], A, B, C, epsilon)
    plt.plot(G*L, energy/L, label = "a = {0}, b = {1}, c={2}, L={3}".format(
        A, B, C, L))
    seps = np.sum(epsilon)
    Gc = find_Gc(A, B, C, L, seps)
    plt.axvline(Gc*L)

def Gofg(g, L, N, A, B, C, seps):
    M = N - L/2
    lamb = 1+B*g*(M-1)
    G = -g/(2*lamb - g*C*seps)
    return G

def find_Gc(A, B, C, L, seps):
    N = L//4
    gamma = np.sqrt(B**2-A*C)
    # gc = -1./(gamma*(L/2-1)) # crit point I checked and got from mathematica
    gc = -1./(gamma*(L/2+3*N-2)) # crit point from my work
    Gc = Gofg(gc, L, N, A, B, C, seps)
    return Gc
    

def plot_energy_boring(L, steps=11):
    N = L//2
    Gmr = 1/(L-N+1)
    Gc = 1/(L+1)
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    print('Moore-read G is {}'.format(Gmr))
    energy = np.array([0 for i in range(steps)])
    G = np.linspace(0.00001, 0.05, num=steps)
    for i in range(steps):
        energy[i] = compute_iom_energy(L, N, G[i], 'hyperbolic',
                epsilon)
    plt.plot(G, energy, label = "Hyperbolic model, L={}".format(L))
    plt.axvline(Gc)
    plt.axvline(Gmr)
    

if __name__ == '__main__':
    # plot_energy(1, 4, -2, 100)
    # plot_energy(1, 4, 2, 100, steps=100)
    # plot_energy(1, 4, 1, 100, steps=100)
    plot_energy_boring(100, steps=100)
    plt.legend()
    plt.xlabel('G')
    plt.ylabel('E')
    plt.show()
    
