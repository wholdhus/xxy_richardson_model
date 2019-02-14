from solve_rg_model import rgk_spectrum, delta_relations
from solve_rg_model import compute_hyperbolic_energy
import numpy as np
import time
import sys
import pandas as pd

def test_rgk():
    L = 2048
    N = 512
    k, rgke = rgk_spectrum(L, 1, 0, peri=False)
    epsilon = rgke
    G = 3.0/L
    if len(sys.argv) > 1:
        g_step = float(sys.argv[1])/L
    else:
        g_step = 0.1/L
    l = L/2
    n = N/2
    energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(L, N, G, epsilon, g_step)
    energies = 8*energies - 2*N

    df = pd.DataFrame({'G': Gs, 'E': energies})
    df.to_csv('results/rgk_energies_{}.csv'.format(g_step))
    Ns = pd.DataFrame({'k': k, 'epsilon': epsilon}, index = k)
    Deltas = pd.DataFrame({'k': k, 'epsilon': epsilon}, index = k)
    for i, G in enumerate(Gs):
        Ns['G'] = nsk[i]
        Deltas['G'] = deltas[i]
    Ns.to_csv('results/rgk_ns_{}.csv'.format(g_step))
    Deltas.to_csv('results/rgk_deltas_{}.csv'.format(g_step))

if __name__ == '__main__':
    start = time.time()
    test_rgk()
    finish = time.time()
    print('Seconds elapsed: {}'.format(finish-start))
