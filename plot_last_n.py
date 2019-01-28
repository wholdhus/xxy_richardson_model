from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
import numpy as np
import matplotlib.pyplot as plt
import sys

L = 100
N = 75
if len(sys.argv) == 3:
    L = int(sys.argv[1])
    N = int(sys.argv[2])

Gc = 1./(L-2*N+1)
Gmr = 1./(L-N+1)
Gprob = -1./(N-L/2+1)
k, epsilon = rgk_spectrum(L, 1, 0)
g_step = 0.005
steps = 50
# Gs = np.linspace(2*Gc, 0.9*Gc, steps)
# Gs = np.linspace(-0.06, -0.04, steps)
Gs = np.linspace(1.2*Gc, 0, steps, dtype=np.float64)
print(Gs)

nsb = np.zeros(steps)
esb = np.zeros(steps)
nsg = np.zeros(steps)
esg = np.zeros(steps)

condsg = np.zeros(steps)
condsb = np.zeros(steps)

G_cross = -1./(N-L/2-1)

for i, G in enumerate(Gs):
    print("")
    print('G = {}'.format(G))
    E_bad, n_bad, delta, s = compute_hyperbolic_energy(L, N, G, epsilon, g_step, return_matrix=True)
    nsb[i] = n_bad[N-1]
    esb[i] = E_bad
    condsb[i] = np.linalg.cond(s)


    E_good, n_good, delta, s = compute_hyperbolic_energy(L, N, G, epsilon, g_step/10, return_matrix=True)
    nsg[i] = n_good[N-1]
    esg[i] = E_good

    condsg[i] = np.linalg.cond(s)


plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.title('Energies')
plt.scatter(Gs, esb, label='g_step = {}'.format(g_step), s=20)
plt.scatter(Gs, esg, label='g_step = {}'.format(g_step/10), s=8)
plt.axvline(Gc)
plt.xticks(Gs, '')

plt.legend()

plt.subplot(3,1,2)
plt.title('n_N')
plt.scatter(Gs, nsb, label= g_step, s=20)
plt.scatter(Gs, nsg, label = g_step/10, s=8)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.axvline(Gc)
plt.xticks(Gs, '')

plt.subplot(3,1,3)
plt.title('log(cond)')
plt.scatter(Gs, np.log10(condsb), label = g_step, s=20)
plt.scatter(Gs, np.log10(condsg), label = g_step/10, s=8)
plt.axvline(Gc)
# plt.axvline(Gprob)
# plt.xticks(Gs,Gs)

plt.show()
