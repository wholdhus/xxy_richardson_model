import numpy as np
import matplotlib.pyplot as plt
from solve_rg_model import rgk_spectrum, compute_hyperbolic_energy, compute_infinite_G
import celluloid

L = 40
N = 30
k, epsilon = rgk_spectrum(L*2, 1, 0)

Einfs = []


print('Doing ground state also')
# Es, _, _, _, _ = compute_hyperbolic_energy(L, N, G, epsilon, 1./L, init_state=None)
Gs, _, Es = compute_infinite_G(L, N, epsilon, .5/L, init_state=None)
# if min(Es) < Es[0] and G < 0:
if 1 != 1:
    print('Woops unstable')
else:
    Egs = Es[-1]
    print('Final gs energy: {}'.format(Egs))

Esd = {}
for i, G in enumerate(Gs):
    Esd[G] = []
    Esd[G] += [Es[i]]

Einfs = Einfs + [Es[-1]]
tries = 200
for i in range(tries):
    # Es, _, _, _, _ = compute_hyperbolic_energy(L, N, G, epsilon, 1./L, init_state='r')
    _, _, Es = compute_infinite_G(L, N, epsilon, .5/L, init_state='r')
    # if Es[-1] < Egs:
    if 1 != 1:
        print('Woops unstable')
    else:
        Einfs = Einfs + [Es[-1]]
        for j, G in enumerate(Gs):
            Esd[G] += [Es[j]]
    print('Did {} of {}'.format(i+1, tries))


# camera = celluloid.Camera(plt.figure(figsize=(12,8)))

# for i in range(len(Gs)):
#    Ess = (Es[i][:]-min(Es[i][:]))/max(Es[i][:]-min(Es[i][:]))
#    plt.hist(Ess, bins=tries//2, color = 'm')
#    plt.show()
lowest_bin = np.zeros(len(Gs))
print(Esd[Gs[-1]])
for i, G in enumerate(Gs):

    h, we = np.histogram(Esd[G], bins=tries//2)
    print(h[0])
    lowest_bin[i] = h[0]

plt.hist(Esd[G], bins=tries//2)
plt.title('G= {}'.format(G))

    # plt.show()

plt.plot(lowest_bin)
plt.show()
