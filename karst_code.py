from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
import numpy as np
import pandas as pd
import sys
import time

start = time.time()

L = int(sys.argv[1])
N = int(sys.argv[2])
g_step = float(sys.argv[3])/L
spectrum = sys.argv[4]

print('Running with params L, N, spectrum = {}, {}, {}'.format(
    L, N, spectrum))
l = int(L/2)
n = int(N/2)
Grg = 1./(l-2*n+1)

# G = -1.5*Grg
g = -500.0
G = g/L

k, epsilon = rgk_spectrum(L, 1, 0)
if spectrum != 'rgk':
    expo = float(spectrum)
    epsilon = k**expo

energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(l, n, G, epsilon, g_step)

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

finish = time.time()

print('Seconds elapsed: {}'.format(finish-start))
