from solve_rg_model import compute_hyperbolic_deltas, compute_iom_energy
import numpy as np
import matplotlib.pyplot as plt

L = 100
N = 60
t1 = 1
t2 = 0

k = np.linspace(-np.pi,0,L)
eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2))**2)
epsilon = eta**2

G_c = 1./(L-2*N+1)

E, n, delta = compute_iom_energy(L, N, G_c, 'hyperbolic', epsilon,
        steps = 100,
        return_delta = True,
        taylor_expand = False)

E10, n10, delta10, _ = compute_hyperbolic_deltas(L, N, G_c, epsilon, gsteps = 100,
        imsteps = 10)
E100, n100, delta100, _ = compute_hyperbolic_deltas(L, N, G_c, epsilon, gsteps = 100,
        imsteps = 10)

print('Energies are {} {} {}'.format(E, E10, E100))
plt.plot(n, label = 'No imaginary stuff')
plt.plot(n10, label = '10 imaginary steps')
plt.plot(n100, label = '100 imaginary steps')
plt.legend()
plt.show()
