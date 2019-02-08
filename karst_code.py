from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
import numpy as np
import pandas as pd
import sys

if len(sys.argv) == 4:
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    g_step = float(sys.argv[3])/L
else:
    L = 20
    N = 13
    g_step = 0.1/L

print('Running with params L, N, steps = {}, {}, {}'.format(
    L, N, g_step))

Grg = 1./(L-2*N+1)

G = 5.*Grg

k, epsilon = rgk_spectrum(L, 1, 0)
epsilon = k**2
epsilon = epsilon + 0.1

energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(L, N, G, epsilon, g_step)


Energy = pd.DataFrame({'G': Gs, 'Energy': energies}, index = Gs)
Ns = pd.DataFrame({'k': k, 'epsilon': epsilon}, index = k)
Deltas = pd.DataFrame({'k': k, 'epsilon': epsilon}, index = k)
for i, G in enumerate(Gs):
    Ns[G] = nsk[i]
    Deltas[G] = deltas[i]

prefix = 'results/L{}_N{}_dg{}_'.format(L, N,  g_step)
Energy.to_csv(prefix + 'energy.csv', index=False)
Ns.to_csv(prefix + 'occupation.csv', index=False)
Deltas.to_csv(prefix + 'deltas.csv', index=False)
