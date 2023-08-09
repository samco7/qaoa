from time import time
from quantum_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from seaborn import color_palette as cp
from qiskit.providers.aer import AerSimulator


def experiment_6(solvers, beta_max=np.pi, gamma_max=np.pi, beta_step=.1, gamma_step=.1, save=True, seed_used=None):
    graph = solvers[0].graph
    exact_val = akmaxsat(graph)[1]
    n_qubits = len(graph.nodes())
    n_layers = solvers[0].n_layers
    shots = solvers[0].shots

    # betas_inner = np.linspace(-1, 1, n_betas_inner)
    # betas_outer = np.linspace(1, np.pi, n_betas_outer//2 + 1)[1:]
    # betas = np.concatenate((-betas_outer[::-1], betas_inner, betas_outer))
    betas_positive = np.arange(0, beta_max, beta_step)
    betas = np.concatenate((-betas_positive[::-1], betas_positive[1:]))
    n_betas = len(betas)

    # gammas_inner = np.linspace(-1, 1, n_gammas_inner)
    # gammas_outer = np.linspace(1, np.pi, n_gammas_outer//2 + 1)[1:]
    # gammas = np.concatenate(((-gammas_outer[::-1], gammas_inner, gammas_outer)))
    gammas_positive = np.arange(0, gamma_max, gamma_step)
    gammas = np.concatenate((-gammas_positive[::-1], gammas_positive[1:]))
    n_gammas = len(gammas)

    X, Y = np.meshgrid(betas, gammas)

    labels = []
    landscapes = []
    progress = tqdm(total=n_betas*n_gammas*len(solvers))
    for solver in solvers:
        labels.append(solver.label)
        landscape = np.empty((n_betas, n_gammas))
        for i in range(n_betas):
            for j in range(n_gammas):
                counts = solver.execute_circuit([betas[i]], [gammas[j]])
                expectation = solver.compute_expectation(counts)
                landscape[i, j] = expectation/exact_val
                progress.update(1)
        landscapes.append(landscape.T)
    progress.close()
    res = {'beta_mesh':X, 'gamma_mesh':Y, 'landscapes':landscapes, 'solvers':solvers,
    'shots':shots, 'n_layers':n_layers, 'n_qubits':n_qubits, 'beta_step':beta_step, 'gamma_step':gamma_step, 'labels':labels}
    # 'n_betas_inner':n_betas_inner, 'n_betas_outer':n_betas_outer,
    # 'n_gammas_inner':n_gammas_inner, 'n_gammas_outer':n_gammas_outer
    info = {'n_qubits':n_qubits, 'n_layers':n_layers, 'beta_step':beta_step, 'gamma_step':gamma_step, 'shots':shots,  'seed':seed_used}

    # 'n_betas_inner':n_betas_inner, 'n_betas_outer':n_betas_outer,
    # 'n_gammas_inner':n_gammas_inner, 'n_gammas_outer':n_gammas_outer,
    if save:
        save_result(res, 'experiment_6')
        save_info(info, 'experiment_6')
    return res


def plot_experiment_6(res, save=True, date=None, cmap='magma'):
    X = res['beta_mesh']
    Y = res['gamma_mesh']
    landscapes = res['landscapes']
    labels = res['labels']

    vmins = np.zeros(len(landscapes))
    vmaxes = np.zeros(len(landscapes))
    for landscape, i in zip(landscapes, range(len(landscapes))):
        vmin = 1
        vmax = 0
        for item in landscape.ravel():
            if item < vmin: vmin = item
            if item > vmax: vmax = item
        vmins[i] = vmin
        vmaxes[i] = vmax

    n_rows = int(np.ceil(len(labels)/2))
    fig, ax = plt.subplots(n_rows, 2, figsize=(8, 3.5*n_rows))
    ax = ax.ravel()

    if n_rows*2 > len(landscapes):
        ax[-1].xaxis.set_visible(False)
        ax[-1].yaxis.set_visible(False)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax[-1].spines[spine].set_visible(False)


    for i in range(len(landscapes)):
        mesh = ax[i].pcolormesh(X, Y, landscapes[i], cmap=cmap, vmin=vmins[i], vmax=vmaxes[i])
        ax[i].set_xlabel(r'$\beta$', fontsize=12)
        ax[i].set_ylabel(r'$\gamma$', fontsize=12)
        ax[i].set_title(labels[i], fontsize=12)
        bar = plt.colorbar(mesh, ax=ax[i])
        bar.set_label(label='Approximation Ratio', fontsize=12)
    plt.suptitle('Single-Layer Parameter Landscape', fontsize=14, y=1)
    plt.tight_layout()
    if save:
        save_plot('experiment_6', date=date)
    plt.show()
