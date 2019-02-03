from solve_rg_model import compute_hyperbolic_energy
from exact_diag import compute_n_exact
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

L = 50
dens = 0.8
N = int(dens*L)
Gc = 1./(L-2*N+1)
l = 5

g_step = 0.001

G_cross = -1./(N-L/2-1)
# Gs = np.linspace(1.05*Gc, 0.95*Gc, l, dtype=np.float64)

Gs = np.linspace(-0.05, -0.04, l-1, dtype=np.float64)
Gs = np.append(Gs, -0.038)

ns1 = [np.zeros(L, np.float64) for i in range(l)]
# ns2 = [np.zeros(L, np.float64) for i in range(l)]
nsd = [np.zeros(L, np.float64) for i in range(l)]

deltas1 = [np.zeros(L, np.float64) for i in range(l)]
# deltas2 = [np.zeros(L, np.float64) for i in range(l)]


print('Predicted critical coupling is G={}'.format(Gc))
print('~~crossover~~ coupling is G={}'.format(G_cross))
print('')
k = np.pi*np.linspace(0, 1, L, dtype=np.float64)
epsilon = -0.5 * np.cos(k) + 0.5

f, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
a = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]]

for i, G in enumerate(Gs):
    print('')
    print('Trying G-Gc = {}'.format(G-Gc))
    E, n, delta, A = compute_hyperbolic_energy(L, N, G, epsilon,
                                               g_step=g_step)
    ns1[i] = n
    deltas1[i] = delta
    _, svs, _ = np.linalg.svd(A)
    print('Max sv: {}'.format(np.max(np.abs(svs))))
    print('Min sv: {}'.format(np.min(np.abs(svs))))
    det = np.linalg.det(A)
    print('Determinant: {}'.format(det))
    sns.heatmap(A, cmap="GnBu", ax=a[i], xticklabels=False, yticklabels=False,
                vmin = -10, vmax = 200)
    a[i].set_title('G-Gc={}'.format(np.round(G-Gc, 4)))

    if G == Gc:
        Ac = A
    elif i == 0:
        A0 = A
    if L < 13:
        E, nsd[i] = compute_n_exact(L, N, G, epsilon)

# sns.heatmap(A0-Ac, cmap="RdBu_r", ax=a[5], xticklabels=False, yticklabels=False, vmin=-25, vmax=25)
# a[i].set_title('A0-Ac')

plt.show()

plt.figure(figsize=(15,9))
plt.subplot(2,1,1)
for i, n in enumerate(ns1):
    g = Gs[i]*L
    if L < 13:
        plt.plot(range(L), nsd[i])
    plt.scatter(range(L), n, label = 'g = {}'.format(np.round(g,4)),
                s=20*(l-i))
plt.ylim(0, 1)

# plt.subplot(2,2,2)
# for i, n in enumerate(ns2):
    # g = Gs[i]*L
    # if L < 13:
        # plt.plot(range(L), nsd[i])
    # plt.scatter(range(L), n, label = 'g = {}'.format(np.round(g,4)),
                # s=3)
# plt.ylim(-2, 2)



plt.subplot(2,1,2)
for i, d in enumerate(deltas1):
    plt.scatter(range(L), d, label = 'G = {}'.format(np.round(Gs[i],4)),
                s=20*(l-i))

# plt.subplot(2,2,4)
# for i, d in enumerate(deltas2):
    # g = Gs[i]*L
    # plt.scatter(range(L), d - deltas1[i], label = 'g = {}'.format(np.round(g,4)),
                # s=3)
plt.legend()

plt.show()
