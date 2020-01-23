import numpy as np
from quspin.basis import boson_basis_1d, spinless_fermion_basis_1d
from quspin.operators import hamiltonian, quantum_operator
from solve_rg_model import rgk_spectrum
# import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def form_basis(L, N):
    basis = boson_basis_1d(L, N, sps=2) # specifying one boson per site
    # print(basis)
    return basis


def construct_pbcs(L, N, epsilon):
    eta = np.sqrt(epsilon)
    basis = boson_basis_1d(L, sps=2)
    creates = [[eta[i], i] for i in range(L)]
    static = [['+', creates]]
    op_dict = {'static': static}
    GammaPlus = quantum_operator(op_dict, basis=basis,
                                 check_herm=False,
                                 check_pcon=False,
                                 check_symm=False)
    # state = (basis.Ns-1)*np.ones(basis.Ns, dtype=int)
    state = np.zeros(basis.Ns)
    state[-1] = 1
    for i in range(N):
        state = GammaPlus.dot(state)
    return state # should be pbcs state


def form_hyperbolic_hamiltonian(L, G, epsilon, N=None, no_kin=True):
    if N is None:
        basis = boson_basis_1d(L, sps=2)
    else:
        basis = form_basis(L, N)
    sqeps = np.sqrt(epsilon)
    if not no_kin:
        hop_vals = -1*G*(np.outer(sqeps, sqeps))
    else:
        hop_vals = np.full((L, L), 1)
    n_vals = np.diag(epsilon)
    hops = []
    for i in range(L):
        for j in range(L):
            new_hop = [[hop_vals[i, j], i, j]]
            hops = hops + new_hop
    pot = [[epsilon[i], i] for i in range(L)]
    if no_kin:
        pot = [[0, i] for i in range(L)]
    static = [['+-', hops], ['n', pot]]
    # H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    op_dict = {'static': static}
    H = quantum_operator(op_dict, basis=basis, check_herm=False, check_symm=False)
    return H, basis


def form_ferm_hamiltonian(L, G, epsilon, N, dd=0):
    NF = int(2*N)
    basis = spinless_fermion_basis_1d(2*L, Nf=int(2*N))
    # if len(epsilon) != 2*L:
    #     epsilon = np.concatenate((epsilon[::-1], epsilon))
    sqeps = np.sqrt(epsilon)
    hop_vals = -1*G*(np.outer(sqeps, sqeps))
    if G <= -999:
        hop_vals = (-1*np.outer(sqeps, sqeps))
    densdens = []
    hops = []
    for i in range(L):
        for j in range(L):
            new_hop = [[hop_vals[i, j], L+i, L-1-i, L-1-j, L+j]]
            if i != j:
                new_dd = [[dd*hop_vals[i, j], L+i, L+j],
                          [dd*hop_vals[i, j], L+i, L-1-j],
                          [dd*hop_vals[i, j], L-1-i, L+j],
                          [dd*hop_vals[i, j], L-1-i, L-1-j]]
                densdens = densdens + new_dd
            else:
                new_dd = [[dd*hop_vals[i, j], L+i, L-1-j],
                          [dd*hop_vals[i, j], L-1-i, L+j]]
            hops = hops + new_hop
    if G > -999:
        kin_neg = [[0.5*epsilon[i], L-1-i] for i in range(L)]
        kin_pos = [[0.5*epsilon[i], L+i] for i in range(L)]
        static = [['++--', hops], ['n', kin_neg], ['n', kin_pos]]
    else:
        static = [['++--', hops]]
    if dd != 0 :
        dds = [['nn',  densdens]]
        static = [['++--', hops], ['nn', densdens]]
    op_dict = {'static': static}
    H = quantum_operator(op_dict, basis=basis, check_herm=False, check_symm=False)
    return H, basis

def charge_gap(L, G, epsilon, N):
    Hn, bn = form_ferm_hamiltonian(L, G, epsilon, N)
    Hm1, bm1 = form_ferm_hamiltonian(L, G, epsilon, N - 0.5)
    Hp1, bp1 = form_ferm_hamiltonian(L, G, epsilon, N + 0.5)
    en, _ = Hn.eigh()
    em1, _ = Hm1.eigh()
    ep1, _ = Hp1.eigh()
    return ep1[0] + em1[0] - 2*en[0]


def hartree_fock_energy(R1d, L, N, H, cons=True, verbose=False):
    # L is number of pairs -> need 2*L fermions
    if cons:
        R = R1d[:-1].reshape((2*L, 2*L))
    else:
        R = R1d.reshape((2*L, 2*L))
    basis1 = spinless_fermion_basis_1d(2*L)
    creates = [None for i in range(2*L)]
    for i in range(N):
        cn = [None for i in range(L)]
        cp = [None for i in range(L)]
        for j in range(L):
            cn[j] = [R[L-1-i, L-1-j], L-1-j]
            cp[j] = [R[L+i, L+j], L+j]
        cn_dict = {'static': [['+', cp]]}
        cp_dict = {'static': [['+', cn]]}
        cnop = quantum_operator(cp_dict, basis=basis1, check_herm=False,
                                check_symm=False,
                                check_pcon=False)
        cpop = quantum_operator(cn_dict, basis=basis1, check_herm=False,
                                check_symm=False,
                                check_pcon=False)
        creates[L-1-i] = cnop # alpha_-k^d
        creates[L+i] = cpop # alpha_k^d
    state = np.zeros(basis1.Ns)
    state[-1] = 1 # starting in the vacuum state
    n = 0
    for i, alpha in enumerate(creates):
        if alpha is not None:
            state = alpha.dot(state)
            n = n + 1
    if cons:
        hf_energy = H.matrix_ele(state, state)/np.dot(state, state) + R1d[-1]*np.abs(1-np.dot(state, state))
    else:
        hf_energy = H.matrix_ele(state, state)/np.dot(state, state)
    if verbose:
        print('There were {} excitations'.format(n))
        print('Hartree fock state')
        inds = np.argwhere(state != 0)
        for i in inds:
            bind = basis1.states[i]
            print(state[i])
            print('times')
            print(basis1.int_to_state(bind))
            print('plus')
        print('Norm:')
        print(np.dot(state, state))
    return float(hf_energy)


def test_pbcs():
    from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
    L = 12
    N = 9
    k = np.pi*np.linspace(0, 1, L, dtype = np.float64)
    epsilon = k**2
    G = -1./(N-1)
    # G = 1./(L-N+1)
    energies, nsk, deltas, G_path, Z = compute_hyperbolic_energy(L, N, G, epsilon, 0.1/L)
    print(energies[-1])
    state = construct_pbcs(L, N, epsilon)
    H = form_hyperbolic_hamiltonian(L, G, epsilon, N=None)
    norm = 1./np.sqrt(np.dot(state, state))
    state = state * norm
    Epbcs = H.matrix_ele(state, state)
    print(Epbcs)
    Eguess = N*np.sum(epsilon)/(N-1)
    print(Eguess)
    print(np.max(np.abs(H.dot(state) - Epbcs * state)))
    Hn = form_hyperbolic_hamiltonian(L, G, epsilon, N=N)
    denergies, _ = Hn.eigh()
    print(np.min(denergies))


def corr_fns(state, start_site, sites, basis):
    corr = np.zeros(len(sites))
    for i, s in enumerate(sites):
        static = [['nn', [[1, start_site, s]]]]
        op_dict = {'static': static}
        O = quantum_operator(op_dict, basis=basis,
                             check_herm=False, check_symm=False,
                             check_pcon=False)
        corr[i] = O.matrix_ele(state, state)
    return corr


def test_hf(L, N):
    R = np.diag(np.ones(2*L))
    R1d = np.append(R.flatten(), 1)
    # R1d = R.flatten()
    k, epsilon = rgk_spectrum(2*L, 1, 0)
    do = 0
    Ehf = []
    Ghf = []
    Ee = []
    G_path = np.linspace(0, 1, 20)*2./(L-2*N+1)
    for i, G in enumerate(G_path):
        if do % 5 == 0:
            Ghf = Ghf + [G]
            H = form_ferm_hamiltonian(L, G, epsilon, N=None)
            # sol = minimize(hartree_fock_energy, R1d, args=(L, N, H))
            sol = minimize(hartree_fock_energy, R1d, args=(L, N, H), method='Powell')
            # sol = minimize(hartree_fock_energy, R1d, args=(L, N, H))
            R1d = sol.x
            EhfHere = hartree_fock_energy(R1d[:-1], L, N, H, verbose=True, cons=False)
            Ehf = Ehf + [EhfHere]

            Hf = form_ferm_hamiltonian(L, G, epsilon, N=N)
            e, v = Hf.eigh()
            EeHere = np.min(e)
            Ee = Ee + [EeHere]
            print('')
            print('G={}'.format(G))
            print('HF energy')
            print(EhfHere)
            print('Exact energy')
            print(Ee)
            print('Correlation energy')
            print(EhfHere - EeHere)
        else:
            print('Skipping')
        do = do + 1
    energies = pd.DataFrame()
    energies['Hartree_Fock'] = Ehf
    energies['Diagonalization'] = Ee
    energies['G'] = Ghf
    energies.to_csv('hf_energies_L{}_N{}.csv'.format(L, N))

"""
Code for calculating spectral functions A(k, omega)
"""
def matrix_elts(k, vs, basis):
    kl = [[1.0, k, k]]
    ol = [['+-', kl]]
    od = {'static': ol}
    op = quantum_operator(od, basis=basis, check_symm=False,
                          check_herm=False)
    l = len(vs[0,:])
    elts = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            elts[i, j] = op.matrix_ele(vs[:,i], vs[:,j])
    return elts


def akw(omega, m_elts, e1s, e2s, eps=10**-6):
    l1 = len(e1s)
    l2 = len(e2s)
    Gs = np.zeros((l1, l2), dtype=np.complex128)
    for i, e1 in enumerate(e1s):
        Gs[i, :] = np.abs(m_elts[i, :])**2/(omega+e1-e2s+1j*eps)
    G = np.sum(Gs)
    return -1*G.imag/np.pi


def nk(L, v, basis):
    nks = np.zeros(L)
    for i in range(L):
        op = quantum_operator({
            'static': [['n', [[1.0, i]]]]
            }, basis=basis)
        melt = op.matrix_ele(v, v)
        print(melt)
        nks[i] = np.real(melt)
    return nks


def constant_op_dict(L, c):
    co = []
    for k in range(L):
        co += [[c, k]]
    const = [['I', co]]
    return {'static': const}


def find_min_ev(operator, L, basis, n=1):
    e, v = operator.eigsh(k=1)
    if e[0] > 0:
        print('Positive eigenvalue: finding a minimum')
        cdict = constant_op_dict(L, np.max(e)/L)
        cop = quantum_operator(cdict, basis=basis)
        e2, v = (cop-operator).eigsh(k=n)
        e = np.max(e) - e2
    if n == 1:
        return e[0], v[:, 0]
    else:
        return e[:n], v[:, :n]


def hf_overlaps(L, v0, dim, basis, kf):
    overlaps = np.zeros(dim, np.complex128)
    nkf = np.zeros(dim)
    for i in range(dim):
        v = np.zeros(dim)
        v[i] = 1.0
        overlaps[i] = np.vdot(v0, v)
        vn = basis.inplace_Op(v, 'n', [kf], 1, np.float64)
        nkf[i] = np.vdot(vn, v)
    # Now to filter with only those that have kf + 1 occupied?
    # nkf = 0.5 + nkf
    return overlaps, nkf

def angular_momentum_op(L, N, basis):
    jd = [[i, i] for i in range(2*L)]
    static = [['n', jd]]
    op = quantum_operator({'static': static}, basis=basis)
    return op

def sqrt_curve(l, a, b, c):
    return a*(l**b) + c


if __name__ == '__main__':
    L = int(input('Max length: '))
    N = int(input('Number of pairs: '))
    g = float(input('Coupling: '))
    ind = int(input('Slater determinant to check overlap with: '))
    power = int(input('Polynomial power: '))

    k, epsilon = rgk_spectrum(2*L, 1, 0)


    cg, _ = charge_gap(L, g, epsilon, N)
    print('Charge gap at L = {}:'.format(L))
    print(cg)

    cgs = np.zeros(5)
    ls = np.arange(L-N-1) + N + 2
    for i, l in enumerate(ls):
        cgs[i], _ = charge_gap(l, g, epsilon, N)
    plt.scatter(1./ls, cgs/ls)
    plt.show()

    more = input('Press Y to continue: ')
    if more == 'Y':

        ls = np.arange(L - N - 1) + N + 2
        print(ls)
        aks = np.zeros(len(ls))

        for i, l in enumerate(ls):
            G = g/l
            print('For L = {}, G = {}'.format(l, G))
            k, epsilon = rgk_spectrum(2*l, 1, 0)
            if g == -999:
                H, basis = form_hyperbolic_hamiltonian(int(l), G, epsilon, N, no_kin=True)
            else:
                H, basis = form_hyperbolic_hamiltonian(int(l), G, epsilon, N, no_kin=False)
                dim = basis.Ns
                # print(dim)
                e, v = H.eigh()
                v0 = v[:, 0]
                kf = N
                overlaps, nkfs = hf_overlaps(L, v0, dim, basis, kf)
                overlaps = np.abs(overlaps)**2
                print('L = {}'.format(l))
                print('Occupation of Kf in {}st excited SD'.format(ind))
                print(nkfs[ind])
                print('Overlap of GS with {}st excited SD times dim(Hspace)'.format(ind))
                print(overlaps[ind]/np.mean(overlaps))
                # aks[i] = overlaps[ind]/np.mean(overlaps)
                #print(overlaps[ind]*len(e))
                #aks[i] = overlaps[ind]*len(e)
                aks[i] = overlaps[ind]
                print('Ground state energy:')
                print(e[ind])
        plt.scatter(ls, aks)
        plt.show()

        print('Fitting ...')
        lf = ls.astype(np.float64)
        p, residuals, rank, singular_values, rcond = np.polyfit(1./lf, aks, power, full=True)

        poly = np.poly1d(p)
        print(poly)
        print(residuals)
        plt.plot(1./ls, poly(1./ls))
        plt.scatter(1./ls, aks)
        plt.show()


        nks = nk(L, v0, basis)
        plt.scatter(range(L), nks)
        plt.show()
