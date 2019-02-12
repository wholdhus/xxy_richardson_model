from celluloid import Camera
from solve_rg_model import rgk_spectrum, delta_relations
from solve_rg_model import compute_hyperbolic_energy
import numpy as np
import matplotlib.pyplot as plt
import sys

def test_rgk():
    L = 2048
    N = 512
    k, rgke = rgk_spectrum(L, 1, 0, start_neg=True)
    epsilon = rgke
    G = 2.3/L
    g_step = 1/L
    energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(L, N, G, epsilon, g_step)
    e1 = np.gradient(energies, G)
    e2 = np.gradient(e2, G)
    e3 = np.gradient(e3, G)
    fig = plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(Gs*L, e2)
    plt.subplot(2,1,2)
    plt.plot(Gs/L, e3)


def examine_deltas():
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    # N = int(0.75*L)
    Grg = 1./(L-2*N+1)
    Gp = -1./(N-L/2-1)
    k, rgke = rgk_spectrum(L, 1, 0, start_neg=False)
    if sys.argv[3] == 'rgk':
        epsilon = rgke
        spectrum = 'rgk'
    else:
        epsilon = k**float(sys.argv[3])
        spectrum = 'k^{}'.format(sys.argv[3])
    G = Grg * 1.2
    print(G)
    g_step = 0.1/L
    if len(sys.argv) > 4:
        g_step = float(sys.argv[4])/L

    print('Params: L, N, spectrum, g_step = {} {} {} {}'.format(L, N, spectrum, g_step))

    # now doing stuff
    energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(L, N, G, epsilon,
                                                             g_step)
    fig = plt.figure(figsize=(12, 8))
    camera = Camera(fig)

    gs = -Gs/(1+Gs*(N-L/2-1))
    der_deltas = np.gradient(deltas, gs, axis=0)
    for i, delta in enumerate(deltas):
        Gi = Gs[i]
        g = gs[i]
        if Gi > Grg:
            color = 'g'
            color2 = 'b'
        else:
            color = 'm'
            color2 = 'r'
        plt.scatter(k, -0.5*delta, color=color, s=4)
        plt.scatter(k, 0.5*g*der_deltas[i], color=color2, s=4)
        plt.scatter(k, nsk[i], color ='black', s=4)
        # iom = -1./2 - delta/2 + g/4*np.sum(Z, axis=1)
        # plt.scatter(k, iom, color=color, s = 8)
        # plt.scatter(k, epsilon*iom, s = 8, color = 'c')
        # plt.axhline(np.dot(epsilon, iom), color = 'y')
        plt.axhline(0, color = 'black')
        camera.snap()
    animation = camera.animate()
    # animation.save('animation.mp4')

    plt.show()

    # doing extra plotz
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    # plt.scatter(Gs, energies)
    lambds = 1./(1+Gs*(N-L/2-1))
    eterm1 = np.zeros(len(Gs))
    eterm2 = np.zeros(len(Gs))
    idots = np.zeros(len(Gs))

    for i, g in enumerate(gs):
        iom = -1./2 - deltas[i]/2 + g/4*np.sum(Z, axis=1)
        eterm1[i] = (1/lambds[i] ) * np.dot(epsilon, iom)
        eterm2[i] = np.sum(epsilon)*(1./2 - 3/4*Gs[i])
        idots[i] = np.dot(epsilon, iom)
    # plt.scatter(Gs, eterm1, color='m', s=4)
    # plt.scatter(Gs, eterm2, color='c', s=4)
    # plt.scatter(Gs, idots, color = 'orange', s=4)
    plt.scatter(Gs, energies, color = 'black', s=4)
    plt.axvline(Grg)
    if G < Gp < 0 or G > Gp > 0:
        plt.axvline(Gp, color='r')

    plt.subplot(2,1,2)
    jumps = [ns[N-1] - ns[N] for ns in nsk]
    plt.scatter(Gs, jumps, s=4)
    plt.axhline(1, ls=':')
    plt.axhline(0, ls=':')
    # plt.ylim(-0.5, 1.5)
    plt.show()

if __name__ == '__main__':
    # examine_deltas()
    test_rgk()
