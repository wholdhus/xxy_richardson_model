from solve_rg_model import compute_hyperbolic_energy
from solve_rg_model import compute_iom_energy
import matplotlib.pyplot as plt
import numpy as np

L = 100
N = 75
k = np.linspace(0,1,L)*np.pi
t1 = 1
t2 = 0
eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2))**2)
epsilon = eta**2

Gc = 1/(L-2*N + 1)
G = 1.5*Gc
Gs = G*np.linspace(0, 1, 50)
holds = np.linspace(0, 0.5, 5)
Es = np.zeros(len(Gs))
Ees = np.zeros(len(Gs))
for j, h in enumerate(holds):
    for i, G in enumerate(Gs):
        Es[i], n, drs, s = compute_hyperbolic_energy(L, N, G, epsilon,
                holdover=h, gstep=0.001)
        Ees[i] = compute_iom_energy(L, N, G, 'hyperbolic',
                epsilon, steps=100, taylor_expand=False,
                return_delta=False, return_n=False)
    plt.plot(Gs, Es, label = h)
    plt.plot(Gs, Ees, label = 'oldway')
plt.axvline(Gc)
plt.legend()
plt.show()

