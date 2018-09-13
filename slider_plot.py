import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation as animation
from solve_rg_model import compute_iom_energy_quad, compute_iom_energy
from pyexact.build_mb_hamiltonian import build_mb_hamiltonian
from pyexact.expected import compute_P

L = 100
fig, ax = plt.subplots(figsize=(12,8))
N = 3*L//4
# N = L//2
# N = L//4
eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
epsilon = eta**2
# k = np.linspace(1, L, L)
# epsilon = k**2
G = 0
print('Computing energies for G={}'.format(G))
E, n = compute_iom_energy(L, N, G, 'hyperbolic', epsilon)

def print_matrix(matrix):
    for row in matrix:
        for i in range(len(row)):
            row[i] = np.round(row[i], 2)
        print(', '.join(map(str, row)))

def compute_n_exact(G, L, N, epsilon):
    sqeps = np.sqrt(epsilon)
    J = -G*np.outer(sqeps, sqeps) + np.diag(epsilon)
    D = np.zeros((L,L), float)
    H = build_mb_hamiltonian(
            J, D, L, N)
    # cdtn = np.linalg.cond(H)
    # print('Condition number {}'.format(cdtn))
    # print_matrix(H)
    w, v = np.linalg.eig(H)
    E = w[0]
    v = v.T[0]
    P = compute_P(v, L, N)
    return E, np.diag(P)


# Ee, ne = compute_n_exact(G, L, N, epsilon)

nk, = plt.plot(n, label='not exact', color = 'black')
# nke, = plt.plot(ne, label='exact', ls='--', color = 'c')
# ndif, = plt.plot(n - ne, label='difference')
plt.legend()

slideraxis = plt.axes([0.2, 0.05, 0.65, 0.03])

# axis for seeing all 3/4 fill behavior
sG = Slider(slideraxis, 'G', 0., 10./L, valinit=0)

# axis for seeing where exact model flickers
# sG = Slider(slideraxis, 'G', 0.28, 0.31, valinit=0)

# axis for seeing where quadratic solutions fail
# sG = Slider(slideraxis, 'G', 0.95, 1.1, valinit=0.9)

def update(val):
    G = -sG.val
    print('Computing energies for G={}'.format(G))
    E, n = compute_iom_energy(L, N, G, 'hyperbolic', epsilon)
    # Ee, ne = compute_n_exact(G, L, N, epsilon)
    # print('Energy diff: {}'.format(Ee-E))
    nk.set_ydata(n)
    # nke.set_ydata(ne)
    fig.canvas.draw_idle()

sG.on_changed(update)
plt.show()

