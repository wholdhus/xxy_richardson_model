import numpy as np
import matplotlib.pyplot as plt
from quspin.basis import boson_basis_1d, spinless_fermion_basis_1d
import quspin.basis as b
from quspin.operators import hamiltonian, quantum_operator, quantum_LinearOperator
from scipy.optimize import minimize
from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
import pandas as pd
import seaborn as sns


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


def construct_sd(L, occs, N=None):
    if N == None:
        basis = spinless_fermion_basis_1d(2*L)
    else:
        basis = spinless_fermion_basis_1d(2*L, 2*N)
    # create_inds = [[1, L+j] for j in occs] j can be + or -
    # creates = [['+', create_inds]]
    # Creator = quantum_operator({'static': creates},
                               # basis=basis,
                               # check_herm=False,
                               # check_pcon=False,
                               # check_symm=False)
    zero = np.zeros(basis.Ns)
    zero[-1] = 1 # this is the zero occupation state
    state = zero
    for i in occs:
        Creator = quantum_operator({'static': [['+', [[1, L+ i]]]]}, basis=basis,
                                   check_herm=False, check_pcon=False, check_symm=False)
        state = Creator.dot(state)
    return state


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
    plt.plot(epsilon)
    sqeps = np.sqrt(epsilon)
    hops = []
    for i in range(L):
        for j in range(L):
            new_hop = [[-1*G*np.sqrt(epsilon[L+i]*epsilon[L+j]), L+i, L-1-i, L-1-j, L+j]]
            hops = hops + new_hop
    kin_neg = [[0.5*epsilon[L-1-i], L-1-i] for i in range(L)]
    kin_pos = [[0.5*epsilon[L+i], L+i] for i in range(L)]
    static = [['++--', hops], ['n', kin_neg], ['n', kin_pos]]
    dynamic = []
    H = quantum_operator({'static': static}, basis=basis)
    return H


def hartree_fock_energy(R1d, L, N, H, verbose=False):
    # L is number of pairs -> need 2*L fermions
    R = R1d.reshape((2*L, 2*L))
    basis1 = spinless_fermion_basis_1d(2*L)
    creates = [None for i in range(2*L)]
    for i in range(L):
        cn = [None for i in range(L)]
        cp = [None for i in range(L)]
        for j in range(L):
            cn[j] = [R[L-1-i, L-1-j], L-1-j]
            cp[j] = [R[L+i, L+j], L+j]
        cn_lst = [['+', cp]]
        cp_lst = [['+', cn]]
        cnop = quantum_operator({'static': cn_lst}, basis=basis1,
                                      check_herm=False,
                                      check_symm=False)
        cpop = quantum_operator({'static': cp_lst}, basis=basis1,
                                      check_herm=False,
                                      check_symm=False)
        creates[L-1-i] = cnop # alpha_-k^d
        creates[L+i] = cpop # alpha_k^d
    state = np.zeros(basis1.Ns)
    state[-1] = 1 # starting in the vacuum state
    n = 0
    while n < N:
        state = creates[L+n].matvec(state)
        state = creates[L-n-1].matvec(state)
        n = n + 1
    hf_energy = H.matrix_ele(state, state)/np.dot(state, state)
    if verbose:
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

def print_state(state, basis): 
    inds = np.argwhere(state != 0)
    for i in inds:
        bind = basis.states[i]
        print(state[i])
        print('times')
        print(basis.int_to_state(bind))
        if i != inds[-1]:
            print('plus')


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
    R1d = R.flatten()
    k, epsilon = rgk_spectrum(2*L, 1, 0)
    do = 0
    Ehf = []
    Ee = []
    # G_path = (np.linspace(0, 1, 20)**2)*1.2/(L-2*N+1)
    Gf = 1.6/(L-2*N+1)
    G_path = Gf * (np.linspace(0, 1, 40))
    R1ds = pd.DataFrame({})
    for i, G in enumerate(G_path):
        print('')
        print('G={}'.format(G))
        H = form_ferm_hamiltonian(L, G, epsilon, N=None)
        if i < 10:
            R1d = R1d + 0.1*(0.5-np.random.rand(4*L**2)) # adding a bit of noise just for fun
        sol = minimize(hartree_fock_energy, R1d, args=(L, N, H))
                       # method='TNC')
        R1d = sol.x
        R1ds[G] = R1d
        EhfHere = hartree_fock_energy(R1d, L, N, H, verbose=True)
        Ehf = Ehf + [EhfHere]

        Hf = form_ferm_hamiltonian(L, G, epsilon, N=N)
        e, v = Hf.eigh()
        EeHere = np.min(e)
        Ee = Ee + [EeHere]
        print('Hf success: {}'.format(sol.success))
        print('HF energy')
        print(EhfHere)
        print('Exact energy')
        print(EeHere)
        print('Correlation energy')
        print(EhfHere - EeHere)
        if G == G_path[-1]:
            sns.heatmap(R1d.reshape(2*L, 2*L), center=0)
            plt.show()
    energies = pd.DataFrame()
    energies['Hartree_Fock'] = Ehf
    energies['Diagonalization'] = Ee
    energies['G'] = G_path
    energies.to_csv('hf_energies_L{}_N{}.csv'.format(L, N))


    plt.scatter(G_path, Ehf, label = 'Hartree-Fock')
    plt.scatter(G_path, Ee, label = 'Diagonalization')
    plt.axvline(1./(L-2*N+1))
    plt.legend()
    plt.show()
    R1ds.to_csv('Rs_{}_{}.csv'.format(L, N))


def n_k(L, state, basis, boson=False):
    if boson:
        ks = range(L)
        ns = np.zeros(L)
    else:
        ks = range(2*L)
        ns = np.zeros(2*L)
    norm = np.dot(state, state)
    for k in ks:
        nk = quantum_operator({'static': [['n', [[1, k]]]]}, basis=basis,
                check_herm=False, check_symm=False)
        ns[k] = nk.matrix_ele(state, state)/norm
    # plt.scatter(ks, ns)
    # plt.show()
    print('N = {}'.format(np.sum(ns)))
    return ns



if __name__ == '__main__':
    import sys
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    G = float(sys.argv[3])
    k, epsilon = rgk_spectrum(2*L, 1, 0)
    H = form_ferm_hamiltonian(L, G, epsilon)

    inds_symm = np.array(range(2*N))-N
    print(inds_symm)

    inds_plus = np.array(range(2*N))-int(N/2+1)
    print(inds_plus)

    inds_minus = -1*inds_plus -1

    symm_state = construct_sd(L, inds_symm)
    plus_state = construct_sd(L, inds_plus)
    minus_state = construct_sd(L, inds_minus)

    basis = spinless_fermion_basis_1d(2*L)


    print('Symmetric state:')
    # print_state(symm_state, basis)
    norm = 1./np.sqrt(np.dot(symm_state, symm_state))
    E_symm = H.matrix_ele(symm_state, symm_state)*norm**2
    print('Energy: {}'.format(E_symm))
    print(np.max(np.abs(H.matvec(symm_state)*norm - E_symm*symm_state*norm)))


    print('Plus state')
    # print_state(plus_state, basis)
    norm = 1./np.sqrt(np.dot(plus_state, plus_state))
    E_plus = H.matrix_ele(plus_state, plus_state)*norm**2
    print('Energy: {}'.format(E_plus))
    print(np.max(np.abs(H.matvec(plus_state)*norm - E_plus*plus_state*norm)))

    print('Minus state')
    # print_state(minus_state, basis)
    norm = 1./np.sqrt(np.dot(minus_state, minus_state))
    E_minus = H.matrix_ele(minus_state, minus_state)*norm**2
    print('Energy: {}'.format(E_minus))
    print(np.max(np.abs(H.matvec(minus_state)*norm - E_minus*minus_state*norm)))

    print('Halfsies')
    state = plus_state + minus_state
    norm = np.sqrt(np.dot(state, state))
    E_h = H.matrix_ele(state, state)/(norm**2)
    print(E_h)
    print(np.max(np.abs(H.matvec(state)/norm - E_h * state/norm)))
    n_k(L, state, basis)

    print('Exacto!')
    H2 = form_ferm_hamiltonian(L, G, epsilon, N=N)
    basis_n = spinless_fermion_basis_1d(2*L, 2*N)
    E, V = H2.eigh()
    E0 = 10
    print('')
    for v in V:
        E = H2.matrix_ele(v, v)/(np.dot(v, v)**2)
        if E <= E0:
            E0 = E
            print('New lowest energy state!')
            print_state(v, basis)
            print('Energy is {}'.format(E))

    print('Exacto2!')
    H3 = form_hyperbolic_hamiltonian(L, G, epsilon, N=N)
    E, V = H3.eigh()
    print(E)
    basis_b = form_basis(L, N)
    print_state(V[0], basis_b)
    n_k(L, V[0], basis_b, boson=True)
