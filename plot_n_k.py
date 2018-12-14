import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solve_rg_model import compute_iom_energy_quad, compute_iom_energy, compute_hyperbolic_energy


def plot_n_quad(L, N, A, B, C, G):
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    energy, denergy, n = compute_iom_energy_quad(
            L, N, -G, A, B, C, epsilon)
    plt.plot(
        n, label = "G = {}".format(round(-G, 3)))


def plot_n_k(L, N, G):
    # eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    # epsilon = eta**2
    t1 = 1
    t2 = 0
    k = np.linspace(-np.pi, 0, L)
    eta = eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2))**2)
    epsilon = eta**2
    E, n = compute_iom_energy(L, N, G, 'hyperbolic', epsilon)
    plt.plot(
            n, label = "G = {}".format(round(-G,3)))
    plt.legend()


def compare_exact(G, L, N, steps, hold):
    t1 = 1
    t2 = 0
    # t1 = 0.1
    # t2 = 1
    k = np.pi*np.linspace(0, 1, L)/L
    eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2))**2)
    epsilon = eta**2

    E, n, delta, _ = compute_hyperbolic_energy(L, N, G, epsilon,
            steps=steps, holdover=hold)
    print('Energy is {}'.format(E))
    plt.plot(k, n, label = '{}, {}'.format(steps, hold))
    n_exact = [0 for i in range(L)]
    if L < 13:
        from exact_diag import compute_n_exact
        print('Attempting to diagonalize ...')
        sqeps = np.sqrt(epsilon)
        E_exact, n_exact = compute_n_exact(L, N, G, epsilon)
        plt.plot(k, n_exact, label = 'Diagonalization', ls = ':')
    plt.xlabel('k')
    plt.ylabel('n_k')
    return n, n_exact, delta, epsilon


def test_holdovers(G, L, N, steps, hmax):
    holds = np.linspace(0, hmax, 5)
    for h in holds:
        compare_exact(G, L, N, steps, h)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    L = int(input('L: '))
    N = int(input('N: '))
    steps = int(input('steps: '))
    try:
        Gc = 1/(L-2*N + 1)
        print('For these values, Gc is {}'.format(Gc))
    except:
        print('Woops: half filling!')
    G = float(input('G: '))
    h = float(input('Holdover fraction: '))

    test_holdovers(G, L, N, steps, h)
    # filename = '{}_{}_{}.csv'.format(
            # L, N, G)
    # n, n_exact, delta, epsilon = compare_exact(G, L, N, steps, h)
    # data = pd.DataFrame(
            # {'n_k quad': n, 'n_k diag': n_exact, 'delta': delta,
                # 'epsilon': epsilon})
    # data.to_csv(filename)
    # plt.show()
