from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
import numpy as np
import pandas as pd
import sys

if len(sys.argv) == 3:
    L = int(sys.argv[1])
    N = int(sys.argv[2])
else:
    L = 20
    N = 13
steps = 50

print('Running with params L, N, steps = {}, {}, {}'.format(
    L, N, steps))

Gc = 1./(L-2*N+1)

Gs = np.linspace(0, 1.2, steps)*Gc
print(Gs)
print('crit G is {}'.format(Gc))

k, epsilon = rgk_spectrum(L, 1, 0, per='anti')
epsilon = epsilon + 0.1

deltas = pd.DataFrame({'k': k, 'epsilon': epsilon})
ns = pd.DataFrame({'k': k, 'epsilon': epsilon})
other = pd.DataFrame({'G': Gs})
energies = np.zeros(steps)
cdns = np.zeros(steps)

for i, G in enumerate(Gs):
    if np.abs(G) < np.abs(0.2*Gc):
        g_step = 1./L
    elif np.abs(G) < np.abs(0.9*Gc):
        g_step = 0.1/L
    else:
        g_step = 0.01/L
    print('')
    print('G = {}'.format(G))
    E, n, delta, A = compute_hyperbolic_energy(L, N, G, epsilon, g_step,
                                               return_matrix=True)
    deltas['G={}'.format(G)] = delta
    ns['G={}'.format(G)] = n
    energies[i] = E
    cdns[i] = np.linalg.cond(A)

file_start = 'results/upside_{}_{}_'.format(L, N)
deltas.to_csv(file_start + 'deltas.csv')
ns.to_csv(file_start + 'nsk.csv')
other['Energy'] = energies
other['Condition numbers'] = cdns
other.to_csv(file_start + 'everything_else.csv')
print('Final energy is {}'.format(E))
print('Finished with L, N, g_step, steps = {}, {}, {}, {}'.format(
    L, N, g_step, steps))
