import numpy as np
import scipy.sparse as sparse
from pyexact.build_mb_hamiltonian import build_mb_hamiltonian
from pyexact.expected import compute_P
from quspin.basis import boson_basis_1d
from quspin.operators import hamiltonian
from solve_rg_model import compute_iom_energy

def compute_n_exact(L, N, G, epsilon):
    sqeps = np.sqrt(epsilon)
    J = -G*np.outer(sqeps, sqeps) + np.diag(epsilon)
    D = np.zeros((L,L), float)
    H = build_mb_hamiltonian(J, D, L, N)
    if L < 13:
        # print(H)
        w, v = np.linalg.eigh(H)
    else:
        w, v = sparse.linalg.eigsh(H, 10)
    minindex = np.argmin(w)
    E = w[minindex]
    v = v.T[minindex]
    P = compute_P(v, L, N)
    return E, np.diag(P)


def form_basis(L, N):
    basis = boson_basis_1d(L, N, sps=2) # specifying one boson per site
    # print(basis)
    return basis


def form_hyperbolic_hamiltonian(L, N, G, epsilon):
    basis = form_basis(L, N)
    sqeps = np.sqrt(epsilon)
    hop_vals = -G*(np.outer(sqeps, sqeps))
    n_vals = np.diag(epsilon)
    hops = []
    for i in range(L):
        for j in range(L):
            # if i != j:
            new_hop = [[hop_vals[i, j], i, j]]
            hops = hops + new_hop
    pot = [[epsilon[i], i] for i in range(L)]
    static = [['+-', hops], ['n', pot]]
    dynamic = []
    H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    return H

def compute_E(L, N, G, epsilon):
    sqeps = np.sqrt(epsilon)
    J = -G*np.outer(sqeps, sqeps) + np.diag(epsilon)
    D = np.zeros((L,L), float)
    H = build_mb_hamiltonian(J, D, L, N)
    w, v = np.linalg.eigh(H)
    return w # returning full spectrum just for funsies


if __name__ == '__main__':
    G = float(input('G: '))
    L = 5
    N = 3*L//4
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    H = form_hyperbolic_hamiltonian(L, N, G, epsilon)
    # print('Here comes H!')
    # print(H.todense())
    print('Here comes the ground state from QuSpin!')
    E, V = H.eigh()
    print(np.min(E))
    
    E0, n0 = compute_n_exact(L, N, G, epsilon)
    print('Here comes the ground state from PyExact')
    print(E0)


    E, n = compute_iom_energy(L, N, G, 'hyperbolic', epsilon)
    print('Here comes the ground state from eigenvalue-based')
    print(E)
