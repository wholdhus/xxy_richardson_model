import numpy as np
import matplotlib.pyplot as plt
from solve_rich_model import compute_iom_energy

if __name__ == '__main__':
    
    steps = 20
    Ls = [20, 30, 50]
    glmax = .005
    colors = ['m', 'c', 'r', 'g', 'y']
    for j in range(3):
        L = Ls[j]
        print('Trying system size {}'.format(L))
        N = L//2
        eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
        epsilon = eta**2
        # Params for hyperbolic case
        A = 1
        B = 4
        C = 1
        gamma = np.sqrt(B^2-A*C)
        ePlus = (gamma - B)//C
        eMinus = -(gamma + B)//C
        Gc = 1.0/(gamma*(N+L/2-1))

        energy = np.array([0 for i in range(steps)])
        dEnergy = np.array([0 for i in range(steps)])
        G = np.linspace(0.00001, glmax * L, steps)
        for i in range(steps):
            energy[i], dEnergy[i] = compute_iom_energy(
                    L, N, G[i], A, B, C, epsilon)
        plt.plot(
            G/L, dEnergy/L, label = "L = {}".format(L), color = colors[j])
        plt.axvline(Gc/L, ls = '-', color = colors[j])
    plt.ylabel('(dE/dG)/L')
    plt.xlabel('G/L')
    plt.legend()
    plt.show()    
