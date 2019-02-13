from solve_rg_model import rgk_spectrum, delta_relations
from solve_rg_model import compute_hyperbolic_energy
import numpy as np
import time
import sys
import pandas as pd
np.set_printoptions(precision=20)


def test_rgk():
    L = 2048
    N = 512
    k, rgke = rgk_spectrum(L, 1, 0, start_neg=True)
    epsilon = rgke
    G = 3.0/L
    if len(sys.argv) > 1:
        g_step = float(sys.argv[1])/L
    else:
        g_step = 0.1/L
    energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(L, N, G, epsilon, g_step)
    energies = 8*energies - 2*N
    e0 = energies/L
    e1 = np.gradient(e0, Gs*L)
    e2 = np.gradient(e1, Gs*L)
    e3 = np.gradient(e2, Gs*L)
    df = pd.DataFrame({'g=GL': Gs*L, 'E': energies, 'dE/dg': e1, 'd2E': e2, 'd3E': e3})
    df.to_csv('results/rgk_energies.csv')

if __name__ == '__main__':
    start = time.time()
    test_rgk()
    finish = time.time()
    print('Seconds elapsed: {}'.format(finish-start))
