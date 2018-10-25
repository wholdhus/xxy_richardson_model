import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solve_rg_model import compute_iom_energy_quad, compute_iom_energy
from exact_diag import compute_n_exact


def plot_n_quad(L, N, A, B, C, G):
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    energy, denergy, n = compute_iom_energy_quad(
            L, N, -G, A, B, C, epsilon)
    plt.plot(
        n, label = "G = {}".format(round(-G, 3)))


def plot_n_k(L, N, G):
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    E, n = compute_iom_energy(L, N, -G, 'hyperbolic', epsilon)
    plt.plot(
            n, label = "G = {}".format(round(-G,3)))
    plt.legend()


def compare_exact(G, L, N):
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    sqeps = np.sqrt(epsilon)
    E_exact, n_exact = compute_n_exact(L, N, G, epsilon)

    E, n, delta = compute_iom_energy(L, N, G, 'hyperbolic', epsilon,
                    return_delta=True)

    plt.plot(n_exact, label = 'n_exact')
    plt.plot(n, label = 'n', ls = ':')
    plt.legend()
    return n, n_exact, delta, epsilon


if __name__ == '__main__':
    G = float(input('G: '))
    L = int(input('L: '))
    N = int(input('N: '))
    filename = '{}_{}_{}.csv'.format(
            L, N, G)
    n, n_exact, delta, epsilon = compare_exact(G, L, N)
    data = pd.DataFrame(
            {'n_k quad': n, 'n_k diag': n_exact, 'delta': delta,
                'epsilon': epsilon})
    data.to_csv(filename)
    plt.show()
