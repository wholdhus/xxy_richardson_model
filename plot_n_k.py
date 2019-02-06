import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solve_rg_model import compute_hyperbolic_energy


def compare_exact(G, L, N, g_step, skip_Grg, start=0.9,
                  use_finite_diff=False):
    t1 = 1
    t2 = 0
    # t1 = 0.1
    # t2 = 1
    k = np.pi*np.linspace(0, 1, L)/L
    eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2))**2)
    epsilon = eta**2

    E, n, delta, A = compute_hyperbolic_energy(L, N, G, epsilon, g_step,
                                               try_g_inv=True,
                                               skip_Grg=skip_Grg,
                                               start=start,
                                               use_finite_diff=use_finite_diff)
    print('Energy is {}'.format(E))
    # plt.scatter(k, n, label = 'New method')
    n_exact = np.zeros(L)
    if L < 13:
        from exact_diag import compute_n_exact
        print('Attempting to diagonalize ...')
        sqeps = np.sqrt(epsilon)
        E_exact, n_exact = compute_n_exact(L, N, G, epsilon)
        # plt.plot(k, n_exact, label = 'Diagonalization', color = 'm')
    # plt.xlabel('k')
    # plt.ylabel('n_k')
    # plt.legend()
    return n, n_exact, delta, epsilon, np.linalg.cond(A)


if __name__ == '__main__':
    import sys
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    g_step = float(sys.argv[3])
    Gp = 1./(1-N+L/2)
    Grg = 1./(L-2*N+1)
    # Gs = -np.linspace(-0.79 * Gp, -1.21*Gp, 8)
    Gs = -np.linspace(-0.9 * Grg, -1.31 * Grg, 5)
    plt.figure(figsize=(12,8))
    print(Grg)
    plt.subplot(2,1,1)
    for i, G in enumerate(Gs):
        print('')
        print(G)
        n, n_e, delta, epsilon, cond = compare_exact(G, L, N, g_step,
                                                     start=0.8,
                                                     skip_Grg=False)
        k = np.linspace(0, 1, L)*np.pi
        plt.scatter(k, n, label = 'G/Grg = {}, losing {} digits'.format(
                    np.round(G/Grg, 4), np.round(np.log10(cond), 4)), s=6)
    plt.legend()
    plt.ylim(-0.2, 1.2)
    plt.axhline(0)
    plt.axhline(1)
    plt.subplot(2,1,2)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Now repeating with finite difference method')
    for i, G in enumerate(Gs):
        print('')
        print(G)
        n, n_e, delta, epsilon, cond = compare_exact(G, L, N, g_step,
                                                     use_finite_diff=True,
                                                     start=0.8,
                                                     skip_Grg=False)
        k = np.linspace(0, 1, L)*np.pi
        plt.scatter(k, n, label = 'G/Grg = {}, losing {} digits'.format(
                    np.round(G/Grg, 4), np.round(np.log10(cond), 4)), s=6)
    plt.legend()
    plt.ylim(-0.2, 1.2)
    plt.axhline(0)
    plt.axhline(1)

    plt.show()
    # Ls = [10, 20, 50, 100, 200]
    # g_step = 0.0005
    # dens = 0.75
    # for L in Ls:
        # N = int(dens*L)
        # Gc = 1/(L-2*N + 1)
        # print('For these values, Gc is {}'.format(Gc))
        # G = 1.1 * Gc
        # n, n_exact, delta, epsilon = compare_exact(G, L, N, g_step)
        # k = np.linspace(0, 1, L)*np.pi
        # plt.scatter(k, n, label = 'L = {}'.format(L))
        # if L < 13:
            # plt.plot(k, n_exact, label = 'Exact for L = {}'.format(L))
    # plt.ylim(-0.2, 1.2)
    # plt.legend()
    # plt.show()
