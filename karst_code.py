from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
import numpy as np
import pandas as pd

L = 20
N = 13
g_step = 0.01/L

Gc = 1./(L-2*N+1)

Gs = np.linspace(0, 1.5, 100)*Gc

print('crit G is {}'.format(Gc))

k, epsilon = rgk_spectrum(L, 1, 0)

deltas = pd.DataFrame({'k': k, 'epsilon': epsilon})
ns = pd.DataFrame({'k': k, 'epsilon': epsilon})

for G in Gs:
    print('')
    print('G = {}'.format(G))
    E, n, delta, A = compute_hyperbolic_energy(L, N, G, epsilon, g_step,
                                               return_matrix=True)
    print('Got stuff')
    deltas['G={}'.format(G)] = delta
    ns['G={}'.format(G)] = n
deltas.to_csv('deltas.csv')
ns.to_csv('ns.csv')
print('Final energy is {}'.format(E))
