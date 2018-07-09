"""Example 2:
Solve the ground state of the RGK Hamiltonian.
"""

import numpy as np
from scipy.linalg import eigh

from solve_rich_model import compute_iom_energy


if __name__ == '__main__':

    # 1. Hamiltonian's parameters.
    L = 14
    N = L//2
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    A = 1
    B = 4
    C = 1
    gamma = np.sqrt(B^2-A*C)
    # My phase transition point:
    Gc = 1.0/(gamma*(N+L/2-1))
    G = Gc
    delta = 0.5

    # 2. Compute ground state and occupation number.
    do_profile = False
    if do_profile:
        import cProfile
        import pstats

        pr = cProfile.Profile()
        pr.enable()
        E, n = compute_iom_energy(L, N, G, model, epsilon)
        pr.disable()

        ps = pstats.Stats(pr).sort_stats('cumulative')
        ps.print_stats(20)
    else:
        E, n = compute_iom_energy(L, N, Gc, A, B, C, epsilon)
        print(f'At the critical point:')
        print(f'E = \n{E}')
        print(f'occupations =\n{n}')
        
        G = Gc + delta
        E2, n2 = compute_iom_energy(L, N, G, A, B, C, epsilon)
        print(f'At G = G_c + {delta}:')
        print(f'E = \n{E2}')
        print(f'occupations =\n{n2}')

        G = Gc - delta
        E3, n3 = compute_iom_energy(L, N, G, A, B, C, epsilon)
        print(f'At G = G_c - {delta}:')
        print(f'E = \n{E3}')
        print(f'occupations =\n{n3}')

    # 3. Check with an exact diagonalization.
    # TODO: figure out how this works and make it work woops
    try:
        from pyexact.build_mb_hamiltonian import build_mb_hamiltonian
        from pyexact.expected import compute_P

        if L > 20:
            raise ValueError('The dimension of the Hilbert space is too big'
                             + 'for ED.')

        print('\nChecking critical IM results against exact diagonalization:')
        J = np.diag(eta**2) - Gc*np.outer(eta, eta)
        D = np.zeros((L, L), np.float64)
        H = build_mb_hamiltonian(J, D, L, N)
        w, v = eigh(H, eigvals=(0, 0))
        w = w[0]
        v = v.T[0]
        P = compute_P(v, L, N)
        print(f'Both energies equal: {np.isclose(E, w)}')
        print(f'All occupation numbers equal: {np.allclose(n, np.diag(P))}')

    except Exception as e:
        print('Could not check against exact diagonalization')
        print(e)
