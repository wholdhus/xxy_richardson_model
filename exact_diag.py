import numpy as np
from quspin.basis import boson_basis_1d, spinless_fermion_basis_1d
import quspin.basis as b
from quspin.operators import hamiltonian, quantum_operator
from scipy.optimize import minimize
from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
import pandas as pd

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


def form_hyperbolic_hamiltonian(L, G, epsilon, N=None):
    if N is None:
        basis = boson_basis_1d(L, sps=2)
    else:
        basis = form_basis(L, N)
        print(basis)
    sqeps = np.sqrt(epsilon)
    hop_vals = -1*G*(np.outer(sqeps, sqeps))
    n_vals = np.diag(epsilon)
    hops = []
    for i in range(L):
        for j in range(L):
            new_hop = [[hop_vals[i, j], i, j]]
            hops = hops + new_hop
    pot = [[epsilon[i], i] for i in range(L)]
    static = [['+-', hops], ['n', pot]]
    dynamic = []
    # H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    op_dict = {'static': static}
    H = quantum_operator(op_dict, basis=basis, check_herm=False, check_symm=False)
    return H


def form_ferm_hamiltonian(L, G, epsilon, N=None):
    if N is None:
        basis = spinless_fermion_basis_1d(2*L)
    else:
        basis = spinless_fermion_basis_1d(2*L, 2*N)
    if len(epsilon) != 2*L:
        epsilon = np.concatenate((epsilon[::-1], epsilon))
    sqeps = np.sqrt(epsilon)
    hop_vals = -1*G*(np.outer(sqeps, sqeps))
    hops = []
    for i in range(L):
        for j in range(L):
            new_hop = [[hop_vals[L+i, L+j], L+i, L-1-i, L-1-j, L+j]]
            hops = hops + new_hop
    kin_neg = [[0.5*epsilon[L-1-i], L-1-i] for i in range(L)]
    kin_pos = [[0.5*epsilon[L+i], L+i] for i in range(L)] 
    # kin = [[epsilon[L+i], L+i] for i in range(L)]
    static = [['++--', hops], ['n', kin_neg], ['n', kin_pos]]
    op_dict = {'static': static}
    H = quantum_operator(op_dict, basis=basis, check_herm=False, check_symm=False)
    return H


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



if __name__ == '__main__':
    import sys
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    test_hf(L, N)
