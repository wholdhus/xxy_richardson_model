import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation as animation
from solve_rg_model import compute_iom_energy, compute_hyperbolic_energy
from exact_diag import compute_n_exact


L = int(input('System length: '))
dens = float(input('Density: '))
N = int(dens*L)
G_p = 1
G_c = 0.5
try:
    G_p = -1./(N-L/2-1)
    print('Problem G is {}'.format(G_p))
    G_c = 1./(L-2*N+1)
    print('Critical G is {}'.format(G_c))
except Exception as e:
    print(e)

fig, ax = plt.subplots(figsize=(12, 8))
t1 = 1
t2 = 0
# t1 = 0.1
# t2 = 1
k = np.pi*np.linspace(0, 1.0, L)
eta = np.sin(k/2)*np.sqrt(t1 + 4*t2*(np.cos(k/2))**2)
epsilon = eta**2
# G = G_c
G = 0
# E, n, delta, A = compute_iom_energy(L, N, G, 'hyperbolic', epsilon,
                                    # return_delta = True, return_n=True,
                                    # taylor_expand=False)
E, n, delta, success = compute_hyperbolic_energy(L, N, G, epsilon,
                                                 g_step=.005, holdover=0,
                                                 try_g_inv=True)
nk, = plt.plot(k, n, label='not exact', color = 'black')

if L < 13:
    Ee, ne = compute_n_exact(L, N, G, epsilon)
    nke, = plt.plot(k, ne, label='exact', ls='--', color = 'c')

plt.legend()

slideraxis = plt.axes([0.2, 0.05, 0.65, 0.03])

G_cross = -1./(N-L/2-1)
print('Weird crossover G is {}'.format(G_cross))
if G_cross <= 0:
    sG = Slider(slideraxis, 'G', 2*G_cross, 0, valinit=G, color = 'black')
else:
    sG = Slider(slideraxis, 'G', 0, 2*G_cross, valinit=G, color = 'black')
plt.axvline(G_cross, color = 'm')
plt.axvline(G_c, color = 'c')

# if N > L//2:
    # sG = Slider(slideraxis, 'G', 1.5*G_c, -1.5*G_c, valinit=G)
# elif N < L//2:
    # sG = Slider(slideraxis, 'G', -1.5*G_c, 1.5*G_c, valinit=G)
# else:
    # print('Woops no handling for half density yet')


def update(val):
    G = sG.val
    E, n, delta, succ = compute_hyperbolic_energy(L, N, G, epsilon,
                                                  g_step=.005, holdover=0,
                                                  try_g_inv=True)
    # nk.set_ydata(delta)
    nk.set_ydata(n)
    if L < 13:
        Ee, ne = compute_n_exact(L, N, G, epsilon)
        nke.set_ydata(ne)
    fig.canvas.draw_idle()

sG.on_changed(update)
plt.show()
