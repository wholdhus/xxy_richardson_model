import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation as animation
from solve_rg_model import compute_iom_energy_quad, compute_iom_energy
from exact_diag import compute_n_exact

L = int(input('System length: '))
dens = float(input('Density: '))
N = int(dens*L)
G_p = -1./(N-L/2-1)
print('Problem G is {}'.format(G_p))
G_c = 1./(L-2*N+1)
print('Critical G is {}'.format(G_c))

fig, ax = plt.subplots(figsize=(12,8))
# N = L//2
# N = L//4
# eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
# epsilon = eta**2
k = np.linspace(0.0, 1.0, L)
epsilon = k**2
G = 0
# print('Computing energies for G={}'.format(G))
E, n = compute_iom_energy(L, N, G, 'hyperbolic', epsilon)

def print_matrix(matrix):
    for row in matrix:
        for i in range(len(row)):
            row[i] = np.round(row[i], 2)
        print(', '.join(map(str, row)))

Ee, ne = compute_n_exact(L, N, G, epsilon)

nk, = plt.plot(n, label='not exact', color = 'black')
if L < 14:
    nke, = plt.plot(ne, label='exact', ls='--', color = 'c')


plt.legend()

slideraxis = plt.axes([0.2, 0.05, 0.65, 0.03])

if N > L//2:
    sG = Slider(slideraxis, 'G', 1.2*G_p, 0, valinit=0)
elif N < L//2:
    sG = Slider(slideraxis, 'G', 0, 1.2*G_p, valinit=0)
else:
    print('Woops no handling for half density yet')

# axis for seeing all 3/4 fill behavior
# sG = Slider(slideraxis, 'G', 0., 100./L, valinit=0)

# axis for seeing where exact model flickers
# sG = Slider(slideraxis, 'G', 0.28, 0.31, valinit=0)

# axis for seeing where quadratic solutions fail
# sG = Slider(slideraxis, 'G', 0.95, 1.1, valinit=0.9)

def update(val):
    G = sG.val
    # print('Computing energies for G={}'.format(G))
    E, n = compute_iom_energy(L, N, G, 'hyperbolic', epsilon)
    if L < 14:
        Ee, ne = compute_n_exact(L, N, G, epsilon)
        if Ee-E > 10**-10:
            print('Energy diff {} at G='.format(Ee-E, G))
    nk.set_ydata(n)
    nke.set_ydata(ne)
    fig.canvas.draw_idle()

sG.on_changed(update)
plt.show()

