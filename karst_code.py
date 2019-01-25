from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
import numpy as np
import pandas as pd

L = 20
N = 13
g_step = 0.01/L
steps = 50

Gc = 1./(L-2*N+1)

Gs = np.linspace(0, 1.5, steps)*Gc

print('crit G is {}'.format(Gc))

k, epsilon = rgk_spectrum(L, 1, 0)

deltas = pd.DataFrame({'k': k, 'epsilon': epsilon})
ns = pd.DataFrame({'k': k, 'epsilon': epsilon})
other = pd.DataFrame({'G': Gs})
energies = np.zeros(steps)
cdns = np.zeros(steps)

for i, G in enumerate(Gs):
    print('')
    print('G = {}'.format(G))
    E, n, delta, A = compute_hyperbolic_energy(L, N, G, epsilon, g_step,
                                               return_matrix=True)
    print('Got stuff')
    deltas['G={}'.format(G)] = delta
    ns['G={}'.format(G)] = n
    energies[i] = N
    cdns[i] = np.linalg.cond(A)

deltas.to_csv('deltas.csv')
ns.to_csv('ns.csv')
other['Energy'] = energies
other['Condition numbers'] = cdns
other.to_csv('everything_else.csv')
print('Final energy is {}'.format(E))
