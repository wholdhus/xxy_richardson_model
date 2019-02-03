from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
import numpy as np
import matplotlib.pyplot as plt
import sys

L = 50
N = 30
g_step = 0.005
if len(sys.argv) == 3:
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    g_step = float(sys.argv[3])

Gc = 1./(L-2*N+1)
Gmr = 1./(L-N+1)
Gprob = -1./(N-L/2+1)
k, epsilon = rgk_spectrum(L, 1, 0)
steps = 20
ends = [Gc, Gmr, Gprob, 0]
print(ends)
Gs = 2.1*np.linspace(np.min(ends), np.max(ends), steps)
# Gs = np.linspace(3.1*Gprob, 0, steps)

# nsb = np.zeros(steps)
# esb = np.zeros(steps)
ns = np.zeros(steps)
jumps = np.zeros(steps)
esg = np.zeros(steps)

condsg = np.zeros(steps)
condsb = np.zeros(steps)

for i, G in enumerate(Gs):
    print("")
    print('G = {}'.format(G))
    E_good, n_good, delta, s = compute_hyperbolic_energy(L, N, G, epsilon, g_step, try_g_inv=True)
    ns[i] = n_good[N-1]
    jumps[i] = n_good[N-1] - n_good[N]

    condsg[i] = np.linalg.cond(s)


plt.figure(figsize=(10,8))

plt.subplot(2,1,1)
plt.title('Energies')
# plt.scatter(Gs, esb, label='g_step = {}'.format(g_step), s=20)
plt.scatter(Gs, esg, label='g_step = {}'.format(g_step), s=8)
plt.axvline(Gc, color = 'g')
plt.axvline(Gprob, color = 'c')
# plt.axvline(Gmr, color = 'r')

plt.legend()

plt.subplot(2,1,2)
plt.title('n_N')
plt.scatter(Gs, ns, label= g_step, s=20)
plt.scatter(Gs, jumps, label = 'jumps', s=8)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.axvline(Gc, color = 'g')
plt.axvline(Gprob, color = 'c')
# plt.axvline(Gmr, color = 'r')


plt.show()
