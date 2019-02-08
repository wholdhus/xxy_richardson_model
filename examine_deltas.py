from celluloid import Camera
from solve_rg_model import rgk_spectrum, delta_relations
from solve_rg_model import compute_hyperbolic_energy
import numpy as np
import matplotlib.pyplot as plt
import sys

L = int(sys.argv[1])
N = int(0.75*L)
Grg = 1./(L-2*N+1)
Gp = -1./(N-L/2-1)
k, rgke = rgk_spectrum(L, 1, 0, start_neg=False)
if len(sys.argv) > 2:
    epsilon = k**float(sys.argv[2])
else:
    epsilon = rgke
G = Grg * 1.51
g_step = 0.1/L
if len(sys.argv) > 3:
    g_step = float(sys.argv[3])/L
energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(L, N, G, epsilon,
                                                         g_step)
fig = plt.figure(figsize=(12, 8))
camera = Camera(fig)

gs = -Gs/(1+Gs*(N-L/2-1))
for i, delta in enumerate(deltas):
    Gi = Gs[i]
    g = gs[i]
    if Gi > Grg:
        color = 'g'
    else:
        color = 'm'
    # plt.subplot(2,1,1)

    # partc = -1*g**2*N*(L-N)* np.ones(L)
    # plt.scatter(k, partc, color = 'black', s=2)
    # part1 = -2*delta
    # plt.scatter(k, part1, color = 'blue', s=2)
    # part2 = g*np.sum(Z, axis=1)*delta - g*np.dot(Z, delta)
    # plt.scatter(k, part2, color = 'c', s=2)
    # part3 = -delta**2
    # plt.scatter(k, part3, color = 'y', s=2)


    # plt.scatter(k, delta, color=color, s=4)
    iom = -1./2 - delta/2 + g/4*np.sum(Z, axis=1)
    plt.scatter(k, iom, color=color, s = 8)
    plt.scatter(k, epsilon*iom, s = 8, color = 'c')
    plt.axhline(np.dot(epsilon, iom), color = 'y')
    plt.axhline(0, color = 'black')
    camera.snap()
animation = camera.animate()
# animation.save('animation.mp4')

plt.show()

# doing extra plotz
plt.figure(figsize=(12,8))
# plt.subplot(2,1,1)
plt.scatter(Gs, energies)
lambds = 1./(1+Gs*(N-L/2-1))
eterm1 = np.zeros(len(Gs))
eterm2 = np.zeros(len(Gs))
idots = np.zeros(len(Gs))

for i, g in enumerate(gs):
    iom = -1./2 - deltas[i]/2 + g/4*np.sum(Z, axis=1)
    eterm1[i] = (1/lambds[i] ) * np.dot(epsilon, iom)
    eterm2[i] = np.sum(epsilon)*(1./2 - 3/4*Gs[i])
    idots[i] = np.dot(epsilon, iom)
plt.subplot(2,1,1)
plt.scatter(Gs, eterm1, color='m')
plt.scatter(Gs, eterm2, color='c')
plt.scatter(Gs, idots, color = 'orange')
plt.axvline(Grg)
if G < Gp:
    plt.axvline(Gp, color='r')

plt.subplot(2,1,2)
plt.scatter(Gs, energies)
plt.show()
