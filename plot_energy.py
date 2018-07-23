import numpy as np
import matplotlib.pyplot as plt
from solve_rg_model import compute_iom_energy_quad

def plot_energies(A, B, C, steps=30):
    Ls = [10, 20, 30]
    glmax = .05
    colors = ['m', 'c', 'r']
    for j in range(4):
        L = Ls[j]
        print('Trying system size {}'.format(L))
        N = L//2
        eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
        epsilon = eta**2
        gamma = np.sqrt(B^2-A*C)
        ePlus = (gamma - B)//C
        eMinus = -(gamma + B)//C
        Gc = 1.0/(gamma*(N+L/2-1))

        energy = np.array([0 for i in range(steps)])
        dEnergy = np.array([0 for i in range(steps)])
        G = np.linspace(0.00001, glmax * L, steps)
        for i in range(steps):
            energy[i], dEnergy[i] = compute_iom_energy_quad(
                    L, N, G[i], A, B, C, epsilon)
        plt.plot(
            G/L, energy/L, label = "L = {}".format(L), color = colors[j])


if __name__ == '__main__':
    
    plt.subplot(3, 1, 1)
    plot_energies(0, 1, 0)
    plt.ylabel('E/L')
    plt.xlabel('G/L')
    plt.legend()

    plt.subplot(3, 1, 2)
    plot_energies(0, 1, 0.5)
    plt.ylabel('E/L')
    plt.xlabel('G/L')
    plt.legend()

    plt.subplot(3, 1, 3)
    plot_energies(0, 1, 1)
    plt.ylabel('E/L')
    plt.xlabel('G/L')
    plt.legend()

    plt.show()    
