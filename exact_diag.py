import numpy as np
import scipy.sparse as sparse
from quspin.basis import boson_basis_1d
from quspin.operators import hamiltonian

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
