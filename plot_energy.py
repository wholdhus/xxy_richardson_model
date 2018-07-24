import numpy as np
import matplotlib.pyplot as plt
from solve_rg_model import compute_iom_energy_quad

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
    gmax = 1
    N = L//4
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    # epsilon = -2*np.cos(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    energy = np.array([0 for i in range(steps)])
    dEnergy = np.array([0 for i in range(steps)])
    G = np.linspace(-gmax/L, gmax/L, steps)
    for i in range(steps):
        energy[i], dEnergy[i] = compute_iom_energy_quad(
            L, N, G[i], A, B, C, epsilon)
    plt.plot(G*L, energy/L, label = "a = {0}, b = {1}, c={2}".format(
        A, B, C))


if __name__ == '__main__':
    plot_energy(1, 2, 1, 100)
    plot_energy(1, 2, -1, 100)
    plot_energy(0, 2, 0, 100)
    plot_energy(1, 0, 0, 100)
    plot_energy(2, 1, 2, 100)
    plt.legend()
    plt.xlabel('G')
    plt.ylabel('E')
    plt.show()
    
