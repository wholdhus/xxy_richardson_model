from celluloid import Camera
from solve_rg_model import rgk_spectrum, delta_relations
from solve_rg_model import compute_hyperbolic_energy
from solve_rg_model import compute_infinite_G
import numpy as np
import matplotlib.pyplot as plt
import sys
from xxy_richardson_gaudin_bethe import bethe
import exact_diag as ed
import time

np.set_printoptions(precision=20)

def do_infinite(L, N):
    # camera = Camera(fig)
    k, epsilon = rgk_spectrum(L, 1, 0, peri=False)
    l = int(L/2)
    n = int(N/2)
    alpha = 1
    if L < 2*N:
        alpha = -1
    epsilon = epsilon * alpha # relationship between epsilon and eta
    G_path, nsk = compute_infinite_G(l, n, epsilon, float(sys.argv[2])/L)
    if alpha > 0:
        jumps = [ns[n-1] - ns[n] for ns in nsk]
    else:
        jumps = [ns[-n] - ns[-(n+1)] for ns in nsk]
    G_path[-1] = 1.1*G_path[-2]
    G_path = G_path * alpha 
    plt.scatter(l*G_path[:-1], jumps[:-1], label='{}, {}'.format(L,N))
    plt.axhline(jumps[-1], ls = ':')
    return jumps[-1]



def compare_bethe(diag=False):
    # doing 3 way test with bethe ansatz and exact diagonalization
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    k, epsilon = rgk_spectrum(2*L, 1, 0, peri=False)
    print(epsilon)
    G =float(sys.argv[3])/(L-2*N+1)
    # G = 1.4/(L-N+1)
    print(G)
    qenergies, qn, deltas, Ges, Z = compute_hyperbolic_energy(L, N, G,
            epsilon, .05/L)
    print('Deltas are:')
    print(deltas[-1])
    # dg=.001/L
    dg = .02/L
    Gmr = 1./(L-N+1)
    if G > Gmr: # want results from 0 to Gmr (>0)
        imscale = .1/L
        # Bethe ansatz results from Gmr -> G
        rE1, iE1, rP1, iP1, er1, Gp1 = bethe.compute_energy(L, N, G, epsilon,
                                                  imscale=imscale, dg=dg, hold=0.0)
        # Bethe ansatz results from Gmr -> 0
        rE2, iE2, rP2, iP2, er2, Gp2 = bethe.compute_energy(L, N, 0, epsilon,
                                                  imscale=imscale, dg=dg, hold=0.0)

        Gp = np.concatenate((Gp2, Gp1[1:]))
        benergies = np.concatenate((rE2, rE1[1:]))
    elif 0 < G < Gmr: # we'll just go from 0 to Gmr
        # imscale = 1./(L**2)
        imscale = 0.01/L
        rE, iE, rP, iP, er, Gp = bethe.compute_energy(L, N, 0, epsilon,
                                                      imscale=imscale, dg=dg, hold=0.0)
        benergies = rE
    else: # we go from Gmr to G
        imscale = .1/L
        rE, iE, rP, iP, er, Gp = bethe.compute_energy(L, N, G, epsilon,
                                                      imscale=imscale, dg=dg, hold=0.5)
        benergies = rE

    l = len(Gp)
    denergies = np.zeros(l)
    if diag:
        for i, Gi in enumerate(Gp):
            print('G = {}'.format(Gi))
            H = ed.form_hyperbolic_hamiltonian(L, N, Gi, epsilon)
            denergies[i] = np.min(H.eigvalsh())
    plt.figure(figsize=(12, 8))
    plt.subplot(2,2,1)
    plt.scatter(Ges, qenergies, label = 'quad', marker = '1')
    plt.scatter(Gp, benergies, label = 'rg', marker = 'x')
    if diag:
        plt.scatter(Gp, denergies, label = 'diag', marker = 'o', s=4,
                    color='r')
    Gmr = 1./(L-N+1)
    Grg = 1./(L-2*N+1)
    Gn = -2./(N-1)
    if np.min(Gp) < Gmr < np.max(Gp):
        plt.axvline(Gmr, color='r')
    if np.min(Gp) < Grg < np.max(Gp):
        plt.axvline(Grg, color = 'g')
    if np.min(Gp) < Gn < np.max(Gp):
        plt.axvline(Gn, color = 'c')
    plt.ylim(0.5*np.min(qenergies), 1.5*np.max(qenergies))
    plt.legend()

    plt.subplot(2,2,2)
    if diag:
        plt.scatter(Gp, benergies-denergies)

    plt.subplot(2,2,3)
    for i in range(N):
        if G > Gmr:
            reals1 = [rP1[j][i] for j in range(len(Gp1))]
            plt.scatter(Gp1, reals1, s=4)
            reals2 = [rP2[j][i] for j in range(len(Gp2))]
            plt.scatter(Gp2, reals2, s=4)
        else:
            reals = [rP[j][i] for j in range(len(Gp))]
            plt.scatter(Gp, reals, s=4)
    if np.min(Gp) < Gmr < np.max(Gp):
        plt.axvline(Gmr, color='r')
    if np.min(Gp) < Grg < np.max(Gp):
        plt.axvline(Grg, color = 'g')
    if np.min(Gp) < Gn < np.max(Gp):
        plt.axvline(Gn, color = 'c')
    plt.ylim(-1, 1)

    plt.subplot(2,2,4)
    for i in range(N):
        if G > Gmr:
            imps1 = [iP1[j][i] for j in range(len(Gp1))]
            plt.scatter(Gp1, imps1, s=4)
            imps2 = [iP2[j][i] for j in range(len(Gp2))]
            plt.scatter(Gp2, imps2, s=4)
        else:
            imps = [iP[j][i] for j in range(len(Gp))]
            plt.scatter(Gp, imps, s=4)
    if np.min(Gp) < Gmr < np.max(Gp):
        plt.axvline(Gmr, color='r')
    if np.min(Gp) < Grg < np.max(Gp):
        plt.axvline(Grg, color = 'g')
    if np.min(Gp) < Gn < np.max(Gp):
        plt.axvline(Gn, color = 'c')
    # plt.ylim(-1, 1)


def examine_deltas():
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    # N = int(0.75*L)
    l = int(L/2)
    n = int(N/2)
    Grg = 1./(l-2*n+1)
    Gp = -1./(n-l/2-1)
    k, rgke = rgk_spectrum(L, 1, 0, peri=False, fix=False)
    spectrum2 = 'RGK'
    if float(sys.argv[3]) == 0:
        epsilon = k**2 - (k**4)/(4*3*2)
        epsilon = epsilon/np.max(epsilon)
        spectrum = 'modified rgk'
        plt.scatter(k, epsilon, marker='o')
        plt.scatter(k, rgke, marker='x')
        plt.show()
    else:
        epsilon = k**float(sys.argv[3])
        spectrum = 'k^{}'.format(sys.argv[3])
    G = Grg * 3.1
    print(G)
    g_step = 0.1/L
    if len(sys.argv) > 4:
        g_step = float(sys.argv[4])/L
    start=0.7

    print('Params: L, N, spectrum, g_step = {} {} {} {}'.format(L, N, spectrum, g_step))

    # now doing stuff
    print('Running with RGK spectrum')
    energies_rgk, nsk_rgk, deltas_rgk, Gs_rgk, Z_rgk = compute_hyperbolic_energy(l, n, G,
                                                                                 rgke, g_step,
                                                                                 start=start)
    print('Running with {} spectrum'.format(spectrum))
    energies, nsk, deltas, Gs, Z = compute_hyperbolic_energy(l, n, G, epsilon,
                                                             g_step, start=start)
    print('Got results. Plotting!')
    fig = plt.figure(figsize=(12,8))
    camera = Camera(fig)
    for i, ds in enumerate(deltas_rgk):
        if Gs_rgk[i] < 0.9*start*Gp:
            print(Gs_rgk[i])
            plt.scatter(k, ds)
    camera.snap()
    camera.animate()
    plt.show()


    gs = -Gs/(1+Gs*(n-l/2-1))
    print('Columnular sum for rgk')
    print(np.sum(Z_rgk, axis=1))

    print('Columnular sum for other')
    print(np.sum(Z, axis=1))

    # doing extra plotz
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    # plt.scatter(Gs, energies)
    lambds = 1./(1+Gs*(n-l/2-1))
    eterm1 = np.zeros(len(Gs))
    eterm2 = np.zeros(len(Gs))
    eterm1r = np.zeros(len(Gs))
    eterm2r = np.zeros(len(Gs))

    ioms = np.zeros((len(Gs), len(k)))
    iomrs = np.zeros((len(Gs), len(k)))

    for i, g in enumerate(gs):
        ioms[i] = -1./2 - deltas[i]/2 + g/4*np.sum(Z, axis=1)
        eterm1[i] = (1/lambds[i] ) * np.dot(epsilon, ioms[i])
        eterm2[i] = np.sum(epsilon)*(1./2 - 3/4*Gs[i])
        iomrs[i] = -1./2 - deltas_rgk[i]/2 + g/4*np.sum(Z_rgk, axis=1)
        eterm1r[i] = (1/lambds[i] ) * np.dot(rgke, iomrs[i])
        eterm2r[i] = np.sum(rgke)*(1./2 - 3/4*Gs[i])
    plt.title('Ground state energy')
    plt.scatter(Gs, energies-energies[0], marker='+',
                label = '{} spectrum'.format(spectrum),
                color = 'c')
    plt.scatter(Gs, energies_rgk - energies_rgk[0], marker='x',
                label = '{} spectrum'.format(spectrum2),
                color = 'm')
    # plt.ylim(0, 1.1*(energies[-3]-energies[0]))
    plt.axvline(Grg)
    if G < Gp < 0 or G > Gp > 0:
        plt.axvline(start*Gp, ls = ':')
        plt.axvline((2-start)*Gp, ls = ':')
        plt.axvline(Gp, color='r')
        plt.xlim(1.1*(2-start)*Gp, 0.9*start*Gp)
    # plt.xlabel('G')
    plt.ylabel('E_0')
    plt.legend()

    plt.subplot(3,1,2)
    plt.title('Elements of r_k as a function of coupling')
    r0s = [iom[0] for iom in ioms]
    r0rs = [iomr[0] for iomr in iomrs]
    rLs = [iom[-1] for iom in ioms]
    rLrs = [iomr[-1] for iomr in iomrs]
    rNs = [iom[n-1] for iom in ioms]
    rNrs = [iomr[n-1] for iomr in iomrs]
    rN1s = [iom[n] for iom in ioms]
    rN1rs = [iomr[n] for iomr in iomrs]
    plt.scatter(Gs, r0s, label = '1st site {}'.format(spectrum), marker = '+', s=4)
    plt.scatter(Gs, r0rs, label = '1st site {}'.format(spectrum2), marker = 'x', s=4)
    plt.scatter(Gs, rNs, label = 'Mth site {}'.format(spectrum), marker = '+', s=4)
    plt.scatter(Gs, rNrs, label = 'Mth {}'.format(spectrum2), marker = 'x', s=4)
    plt.scatter(Gs, rN1s, label = 'M+1th site {}'.format(spectrum), marker = '+', s=4)
    plt.scatter(Gs, rN1rs, label = 'M+1th {}'.format(spectrum2), marker = 'x', s=4)
    plt.scatter(Gs, rLs, label = 'Lth site {}'.format(spectrum), marker = '+', s=4)
    plt.scatter(Gs, rLrs, label = 'Lth site {}'.format(spectrum2), marker = 'x', s=4)
    plt.axvline(Grg)
    if G < Gp < 0 or G > Gp > 0:
        plt.axvline(Gp, color='r')
        plt.xlim(1.1*(2-start)*Gp, 0.9*start*Gp)
    plt.xlabel('G')
    plt.ylabel('r_k(G)')
    plt.legend()

    plt.subplot(3,1,3)
    jumps = [ns[n-1] - ns[n] for ns in nsk]
    jr = [nr[n-1] - nr[n] for nr in nsk_rgk]
    plt.scatter(Gs*l, jumps, marker='o', label = '{} spectrum'.format(spectrum))
    plt.scatter(Gs*l, jr, marker='x', label = 'RGK spectrum')
    # plt.axhline(1, ls=':')
    # plt.axhline(0, ls=':')
    plt.axvline(Grg*l)
    if G < Gp < 0 or G > Gp > 0:
        plt.axvline(Gp*l, color='r')
    plt.ylim(-0.5, 1.5)
    plt.legend()

if __name__ == '__main__':
    import pandas as pd
    start = time.time()
    # examine_deltas()
    # compare_bethe()
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    Ls = np.array([600, 1200, 2400, 4800, 12000])
    jumps = np.zeros(len(Ls))
    for i, L in enumerate(Ls):
        dens = float(sys.argv[1])
        N = dens*L
        jumps[i] = do_infinite(L, N)
    finish = time.time()
    print('Seconds elapsed: {}'.format(finish-start))
    plt.legend()
    plt.subplot(2,1,2)
    plt.scatter(1./Ls, jumps)
    plt.xlim(0, 1./Ls[0])
    plt.ylim(0.85, 1.0)
    plt.show()

    df = pd.DataFrame({'L': Ls, 'Zstar': jumps})
    df.to_csv('results/infinites_{}.csv'.format(int(dens*100)))
