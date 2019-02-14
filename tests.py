from celluloid import Camera
from solve_rg_model import rgk_spectrum, delta_relations
from solve_rg_model import compute_hyperbolic_energy
import numpy as np
import matplotlib.pyplot as plt
import sys
from xxy_richardson_gaudin_bethe import bethe
import exact_diag as ed
import time

np.set_printoptions(precision=20)

def compare_bethe():
    # doing 3 way test with bethe ansatz and exact diagonalization
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    k, epsilon = rgk_spectrum(L, 1, 0, peri=False)

    G =float(sys.argv[3])/(L-2*N+1)
    # G = 1.4/(L-N+1)
    print(G)
    qenergies, qn, deltas, Ges, Z = compute_hyperbolic_energy(L, N, G, epsilon, .1/L)
    print('Deltas are:')
    print(deltas[-1])
    # imscale = 10**-6
    # imscale=5*10**-3
    # dg = 10**-3
    # imscale = 0.1
    # imscale = .1
    # imscale = .01
    imscale = 1./L
    # dg = 10**-3
    dg=.1/L
    re, ie, rp, ip, er = bethe.compute_energy(L, N, G, epsilon,
                                              imscale=imscale, dg=dg)
    print(dg)
    print('Real pairs of pairons:')
    print(rp)
    print('Im parts of pairons:')
    print(ip)
    Gs = Ges
    l = len(Gs)
    Gmr = 1./(L-N+1)
    print('Gmr = {}'.format(Gmr))
    print('')
    benergies = np.zeros(l)
    denergies = np.zeros(l)
    for i, Gi in enumerate(Gs):
        print('G = {}'.format(Gi))
        # if 0.9*Gmr < Gi <= Gmr:
            # print('Close: gonna use smaller imscale and steps')
            # # imscale = 10**-4
            # imscale = 1
            # dg = 10**-4
        # elif Gmr < Gi < 1.1*Gmr:
            # print('Close and above')
            # # imscale = 10**-3
            # imscale=1
            # dg = 10**-4
        # else:
            # imscale = 5*10**-3
            # dg = 10**-3
        print(dg)
        re, ie, rp, ip, er = bethe.compute_energy(L, N, Gi, epsilon,
                        imscale=imscale, dg=dg, hold=0.4)
        if np.abs(re) <= 10**5:
            benergies[i] = re
        else:
            benergies[i] = np.nan
        H = ed.form_hyperbolic_hamiltonian(L, N, Gi, epsilon)
        denergies[i] = np.min(H.eigvalsh())
        print('Pairons:')
        print(rp + 1j*ip)
        print('')
    plt.figure(figsize=(12, 8))
    plt.subplot(2,1,1)
    plt.scatter(Ges, qenergies, label = 'quad', marker = '1')
    # plt.scatter(Gs, benergies, label = 'rg', marker = 'x')
    plt.scatter(Gs, denergies, label = 'diag', marker = 'o', s=4,
                color='r')
    # plt.axvline(1./(L-N+1))
    # plt.axvline(1./(L-2*N+1))
    # plt.ylim(-10, 50)
    plt.legend()

    plt.subplot(2,1,2)
    plt.scatter(Gs, qenergies - denergies, marker='1', label='quad-diag')
    plt.scatter(Gs, benergies - denergies, marker='x', label='bethe-diag')
    plt.ylim(-2*10**-12, 2*10**-12)
    plt.legend()



def test_rgk():
    import pandas as pd
    L = 2048
    N = 512
    k, rgke = rgk_spectrum(L, 1, 0, start_neg=True)
    epsilon = rgke
    l = int(L/2)
    n = int(N/2)
    # G = 3.0/L
    G = 1.5/(l-2*n+1)
    g_step = .1/L
    energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(l, n, G, epsilon, g_step)
    energies = 8*energies - 2*N
    e0 = energies/L
    e1 = np.gradient(e0, Gs*L)
    e2 = np.gradient(e1, Gs*L)
    e3 = np.gradient(e2, Gs*L)
    df = pd.DataFrame({'g=GL': Gs*L, 'E': energies, 'dE/dg': e1, 'd2E': e2, 'd3E': e3})
    df.to_csv('results/rgk_energies.csv')
    fig = plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.scatter(Gs*L, e2)
    plt.subplot(3,1,2)
    plt.scatter(Gs*L, e3)
    plt.subplot(3,1,3)
    plt.scatter(Gs*L, e0)
    plt.show()


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

if __name__ == '__main__':
    start = time.time()
    # examine_deltas()
    # compare_bethe()
    test_rgk()
    finish = time.time()
    print('Seconds elapsed: {}'.format(finish-start))
    plt.show()
