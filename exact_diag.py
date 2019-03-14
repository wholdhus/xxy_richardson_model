import numpy as np
import scipy.sparse as sparse
from quspin.basis import boson_basis_1d, spinless_fermion_basis_1d
import quspin.basis as b
from quspin.operators import hamiltonian, quantum_operator

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
        epsilon = np.concatenate((epsilon, epsilon[::-1]))
    print(epsilon)
    sqeps = np.sqrt(epsilon)
    hop_vals = G*(np.outer(sqeps, sqeps))
    hops = []
    for i in range(L):
        for j in range(L):
            new_hop = [[hop_vals[i, j], i, 2*L-(i+1), j, 2*L-(j+1)]]
            hops = hops + new_hop
    pot = [[0.5*epsilon[i], i] for i in range(L)]
    potpos = [[0.5*epsilon[2*L-(i+1)], 2*L-(i+1)] for i in range(L)] 
    static = [['++--', hops], ['n', pot], ['n', potpos]]
    op_dict = {'static': static}
    H = quantum_operator(op_dict, basis=basis, check_herm=False, check_symm=False)
    return H


def hartree_fock_energy(R1d, L, N, H):
    R = R1d.reshape((L, L))
    # basis1 = boson_basis_1d(L, sps=2)
    basis1 = spinless_fermion_basis_1d(L)
    state = np.zeros(basis1.Ns, dtype=np.complex128)
    state[-1] = 1
    creates = [None for i in range(L)]
    for i in range(N):
        create = []
        for j in range(L):
            create = create + [[R[i,j], j]]
        create_dict = {'static': [['+', create]]}
        qop = quantum_operator(create_dict, basis=basis1, check_herm=False,
                               check_symm=False,
                               check_pcon=False)
        creates[i] = qop
    for alpha in creates:
        if alpha is not None:
            state = alpha.dot(state)
    hf_energy = H.matrix_ele(state, state)/np.dot(state, state)
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
    from scipy.optimize import minimize
    from solve_rg_model import compute_hyperbolic_energy, rgk_spectrum
    R = np.diag(np.ones(L))
    R1d = R.flatten()
    print(len(R1d))
    Gf1 = 1.5/(L-N+1)
    Gf2 = 1.5/(L-2*N+1)
    k, epsilon = rgk_spectrum(2*L, 1, 0)
    energies1, nsk, deltas, G_path1, Z = compute_hyperbolic_energy(L, N, Gf1, epsilon, 0.1/L)
    energies2, nsk, deltas, G_path2, Z = compute_hyperbolic_energy(L, N, Gf2, epsilon, 0.1/L)
    energies = np.concatenate((energies1, energies2[1:]))
    energies = np.sort(energies)[::-1]
    G_path = np.concatenate((G_path1, G_path2[1:]))
    G_path = np.sort(G_path)
    do = 0
    Ehf = []
    Ghf = []
    Ee = []
    for i, G in enumerate(G_path):
        if do % 10 == 0:
            Ghf = Ghf + [G]
            H = form_hyperbolic_hamiltonian(L, G, epsilon, N=None)
            sol = minimize(hartree_fock_energy, R1d, args=(L, N, H))
            R1d = sol.x
            Ehf = Ehf + [hartree_fock_energy(R1d, L, N, H)]
            print('')
            print('G={}'.format(G))
            print('HF energy')
            print(Ehf[-1])
            print('Exact energy')
            print(energies[i])
            print('Correlation energy')
            print(Ehf[-1] - energies[i])
            # Hf = form_hyperbolic_hamiltonian(L, G, epsilon, N=N)
            Hf = form_ferm_hamiltonian(L, G, epsilon, N=N)
            e, v = Hf.eigh()
            Ee = Ee + [np.min(e)]
            print('Difference from diag')
            print(Ee[-1] - energies[i])
        else:
            print('Skipping')
        do = do + 1
    import matplotlib.pyplot as plt
    plt.scatter(G_path, energies, marker = 'x')
    plt.scatter(Ghf, Ee, marker = '1')
    plt.scatter(Ghf, Ehf, marker = 'o')
    plt.scatter([-1./(N-1)], [N*np.sum(epsilon)/(N-1)], marker = 'x')

    # plt.ylim(0, 1.1*N*np.sum(epsilon)/(N-1))
    plt.show()


if __name__ == '__main__':
    # test_pbcs()
    test_hf(4, 3)
