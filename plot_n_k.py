import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from solve_rg_model import compute_iom_energy_quad, compute_iom_energy

def plot_n_quad(A, B, C, g, L):
    N = L//2
    eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    epsilon = eta**2
    energy, denergy, n = compute_iom_energy_quad(
            L, N, g, A, B, C, epsilon)
    plt.plot(
        n, label = "L = {}".format(L))


def plot_n_k(L):
    N = L//2
    k = 2*np.pi*np.linspace(1., L, L)/L
    epsilon = k**2
    # eta = np.sin(np.linspace(1, 2*L-1, L)*np.pi/(4*L))
    # epsilon = eta**2
    # Gmr = 1/(L-N+1)
    # G = np.linspace(0.001, Gmr+0.004, 5)
    G = np.linspace(0.001, 0.4)
    plt.subplot(2, 1, 1)
    for g in G:
        E, n = compute_iom_energy(L, N, g, 'hyperbolic', epsilon)
        plt.plot(
                n[:-L//4], label = "G = {}".format(g))
    # plt.legend()
    plt.subplot(2, 1, 2)
    for g in G:
        E, n = compute_iom_energy(L, N, -g, 'hyperbolic', epsilon)
        plt.plot(
                n[:-L//4], label = "G = {}".format(-g))
    # plt.legend()


if __name__ == '__main__':
    # plot_n_quad(4., 2., 1., 0.05, 50)
    plot_n_k(500)
    # plt.legend()
    plt.show() 
