import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solve_rg_model import compute_iom_energy_quad, compute_iom_energy
from pyexact.build_mb_hamiltonian import build_mb_hamiltonian
from pyexact.expected import compute_P


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
    J = -G*np.outer(sqeps, sqeps)+np.diag(epsilon)
    D = np.zeros((L, L), np.float64)
    H = build_mb_hamiltonian(
            J, D, L, N)
    w, v = np.linalg.eig(H)
    v = v.T[0]
    P = compute_P(v, L, N)
    n_exact = np.diag(P)

    E, n, delta = compute_iom_energy(L, N, G, 'hyperbolic', epsilon,
                    return_delta=True)

    plt.plot(n_exact, label = 'n_exact')
    plt.plot(n, label = 'n')
    plt.legend()
    return n, n_exact, delta


if __name__ == '__main__':
    G = float(input('G: '))
    L = int(input('L: '))
    N = int(input('N: '))
    filename = '{}_{}_{}.csv'.format(
            L, N, G)
    n, n_exact, delta = compare_exact(G, L, N)
    data = pd.DataFrame(
            {'n_k quad': n, 'n_k diag': n_exact, 'delta': delta})
    data.to_csv(filename)
    plt.show()
