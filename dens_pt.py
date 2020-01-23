from solve_rg_model import rgk_spectrum, compute_hyperbolic_energy
import numpy as np
import matplotlib.pyplot as plt

L = int(input('L: '))
g = float(input('g = GL: '))
G = g/L

Ns = np.arange(1, L+1)

k, epsilon = rgk_spectrum(2*L, 1.0, 0)

es = []
Nss = [] # may end up skipping some stuff
for N in Ns:
    if N != L/2:
        print('Doing with N = {}'.format(N))
        e, n, d, gp, z = compute_hyperbolic_energy(L, N, G, epsilon,
                                                   0.01)
        es += [e[-1]]
        Nss += [N]

plt.scatter(Nss, es)
plt.show()

de = np.gradient(es, Nss)
d2e = np.gradient(de, Nss)
d3e = np.gradient(d2e, Nss)
plt.subplot(3,1,1)
plt.scatter(Nss, de)
plt.subplot(3,1,2)
plt.scatter(Nss, d2e)
plt.subplot(3,1,3)
plt.scatter(Nss, d3e)
plt.show()
