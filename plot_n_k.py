import numpy as np
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


def compare_exact(G, L):
    N = 3*L//4
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

    E, n = compute_iom_energy(L, N, G, 'hyperbolic', epsilon)

    print('Max difference between n_k values is {}'.format(
        np.max(n_exact - n)))
    print('N = {}'.format(sum(n)))
    plt.plot(n_exact, label = 'n_exact')
    plt.plot(n, label = 'n')
    plt.legend()


if __name__ == '__main__':
    compare_exact(-0.1, 12)
    plt.show()
