from solve_rg_model import rgk_spectrum, delta_relations
from solve_rg_model import compute_hyperbolic_energy
import numpy as np
import time
import sys
import pandas as pd

def do_rgk():
    L = 2048
    N = 512
    k, rgke = rgk_spectrum(L, 1, 0, peri=False)
    epsilon = rgke
    if len(sys.argv) > 1:
        g_step = float(sys.argv[1])/L
    else:
        g_step = 0.1/L
    print('Parameters: {} {} {}'.format(L, N, g_step))
    l = int(L/2)
    n = int(N/2)
    G = 3.0/l
    energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(l, n, G, epsilon, g_step)
    energies = 8*energies - 2*(N-L/2)

    df = pd.DataFrame({'G': Gs, 'E': energies})
    df.to_csv('RgkResults/rgk_energies_{}_{}.csv'.format(G, np.round(g_step, 2)))
    Ns = pd.DataFrame({'k': k, 'epsilon': epsilon}, index = k)
    Deltas = pd.DataFrame({'k': k, 'epsilon': epsilon}, index = k)
    for i, G in enumerate(Gs):
        Ns['G'] = nsk[i]
        Deltas['G'] = deltas[i]
    Ns.to_csv('RgkResults/rgk_ns_{}_{}.csv'.format(G, np.round(g_step, 2)))
    Deltas.to_csv('RgkResults/rgk_deltas_{}_{}.csv'.format(G, np.round(g_step, 2)))

if __name__ == '__main__':
    start = time.time()
    do_rgk()
    finish = time.time()
    print('Seconds elapsed: {}'.format(finish-start))
