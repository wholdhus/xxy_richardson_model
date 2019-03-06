from solve_rg_model import rgk_spectrum, delta_relations
from solve_rg_model import compute_hyperbolic_energy
from solve_rg_model import compute_infinite_G
import numpy as np
import sys
import time

np.set_printoptions(precision=20)

def do_infinite(L, N, step):
    # camera = Camera(fig)
    k, epsilon = rgk_spectrum(L, 1, 0, peri=False)
    l = int(L/2)
    n = int(N/2)
    alpha = 1
    if L < 2*N:
        alpha = -1
    epsilon = epsilon * alpha # relationship between epsilon and eta
    G_path, nsk = compute_infinite_G(l, n, epsilon, step/L)
    if alpha > 0:
        jumps = [ns[n-1] - ns[n] for ns in nsk]
    else:
        jumps = [ns[-n] - ns[-(n+1)] for ns in nsk]
    G_path[-1] = 1.1*G_path[-2]
    G_path = G_path * alpha
    return jumps[-1]


if __name__ == '__main__':
    import pandas as pd
    start = time.time()
    Ls = np.array([600, 1200, 2400, 4800, 12000])
    jumps = np.zeros(len(Ls))
    dens = float(sys.argv[1])
    steps = float(sys.argv[2])
    for i, L in enumerate(Ls):
        N = dens*L
        jumps[i] = do_infinite(L, N, steps)
    finish = time.time()
    print('Seconds elapsed: {}'.format(finish-start))
    df = pd.DataFrame({'L': Ls, 'Zstar': jumps})
    df.to_csv('results/infinites_{}.csv'.format(int(dens*100)))
