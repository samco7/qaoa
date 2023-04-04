from time import time
from quantum_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from qiskit.providers.aer import AerSimulator


def experiment_1(n_qubits, n_layers, n_trials, epsilon, backend=None, shots=512, save=True):
    reg_list = []
    pole_distances = []
    progress = tqdm(total=n_trials)
    for i in range(n_trials):
        graph = random_graph(n_qubits)
        exact_obj = akmaxsat(graph)[1]

        bmz_qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ', epsilon=epsilon, backend=backend, shots=shots)
        bmz_qaoa._QAOASolver__graph = graph
        bits, obj, angles = bmz_qaoa._QAOASolver__solve_relaxed()
        bmz_qaoa._QAOASolver__set_thetas()
        reg_list.append(bmz_qaoa._QAOASolver__n_regularized)

        a = np.fmin(np.abs(angles), np.abs(angles - np.pi))
        b = np.fmin(a, np.abs(angles - 2*np.pi))
        pole_distances.append(b)
        progress.update(1)
    progress.close()

    res = {'regularizations':reg_list, 'pole_distances':pole_distances, 'epsilon':epsilon, 'shots':shots, 'n_layers':n_layers, 'n_qubits':n_qubits, 'n_trials':n_trials}
    info = {'n_qubits':n_qubits, 'n_layers':n_layers,'n_trials':n_trials, 'epsilon':epsilon, 'shots':shots}
    if save:
        save_result(res, 'experiment_1')
        save_info(info, 'experiment_1')
    return res


def plot_experiment_1(res, save=True, date=None):
    reg_list = res['regularizations']
    pole_distances = res['pole_distances']
    n_trials = res['n_trials']
    epsilon = res['epsilon']

    fig, ax = plt.subplots(1, 2, figsize=(9, 2.5), width_ratios=[.1, .9])
    ax = ax.ravel()

    ax[0].boxplot(reg_list, vert=True)
    ax[0].set_ylabel('Number of angles regularized', fontsize=12)
    ax[0].set_xticks([])
    ax[0].set_title('(a)', x=.2, y=1, fontsize=12)
    ax[0].tick_params(axis='both', which='both', labelsize=12)
    ax[0].set_yticks(np.arange(0, max(reg_list) + 1, 1))


    ax[1].plot([.5, n_trials + .5], [np.pi/4, np.pi/4], ':k')
    ax[1].plot([.5, n_trials + .5], [epsilon, epsilon], linestyle=':', color='royalblue')
    ax[1].boxplot(pole_distances)
    ax[1].set_ylim(-.02, np.pi/2 + .02)
    ax[1].set_xticks([])
    ax[1].set_yticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2], ['0', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{3\pi}{8}$', r'$\frac{\pi}{2}$'])
    ax[1].set_ylabel('Distance to nearest pole', fontsize=12)
    ax[1].set_title('(b)', x=.025, y=1, fontsize=12)
    ax[1].tick_params(axis='both', which='both', labelsize=12)


    plt.tight_layout()
    if save:
        save_plot('experiment_1', date=date)
    plt.show()


# def experiment_1(n_qubits, n_layers, n_trials, epsilons, backend=None, shots=512, save=True):
#     expectations = np.zeros((n_trials, len(epsilons)))
#     vals = np.copy(expectations)
#     progress = tqdm(total=n_trials*len(epsilons))
#     for i in range(len(epsilons)):
#         counter = 0
#         for j in range(n_trials):
#             graph = random_graph(n_qubits)
#             exact_val = akmaxsat(graph)[1]
#             qaoa = QAOASolver(n_layers=n_layers, warm_start_method='BMZ', epsilon=epsilons[i], backend=backend, shots=shots)
#             res = qaoa.solve(graph)
#             expectations[counter, i] = res.expectation/exact_val
#             vals[counter, i] = res.obj/exact_val
#             counter += 1
#             progress.update(1)
#     progress.close()

#     res = {'expectations':expectations, 'vals':vals, 'shots':shots, 'n_layers':n_layers, 'n_qubits':n_qubits, 'n_trials':n_trials, 'epsilons':epsilons}
#     info = {'n_qubits':n_qubits, 'n_layers':n_layers,'n_trials':n_trials, 'epsilons':epsilons, 'shots':shots}
#     if save:
#         save_result(res, 'experiment_1')
#         save_info(info, 'experiment_1')
#     return res


# def plot_experiment_1(res, save=True):
#     expectations, vals = res['expectations'], res['vals']
#     epsilons = res['epsilons']
#     epsilon_strings = [str(round(val, 4)) for val in epsilons]

#     fig, ax = plt.subplots(2, 1, figsize=(8, 10))
#     ax[0].boxplot(expectations, sym='.', positions=range(len(epsilons)),
#     flierprops=dict(markeredgecolor='k'),
#     medianprops=dict(color='steelblue'),
#     boxprops=dict(color='k'),
#     capprops=dict(color='k'),
#     whiskerprops=dict(color='k'))
#     ax[0].set_xticks(range(len(epsilons)), epsilon_strings)
#     ax[0].set_xlabel(r'$\epsilon$ value', fontsize=12)
#     ax[0].set_ylabel('Circuit expectation', fontsize=12)
#     plt.grid(False, axis='x')

#     ax[1].boxplot(vals, sym='.', positions = range(len(epsilons)),
#     flierprops=dict(markeredgecolor='k'),
#     medianprops=dict(color='steelblue'),
#     boxprops=dict(color='k'),
#     capprops=dict(color='k'),
#     whiskerprops=dict(color='k'))
#     ax[1].set_xticks(range(len(epsilons)), epsilon_strings)
#     ax[1].set_xlabel(r'$\epsilon$ value', fontsize=12)
#     ax[1].set_ylabel('Best sampled cut', fontsize=12)
#     plt.grid(False, axis='x')
#     plt.tight_layout()
#     plt.suptitle(f'Experiment 1', fontsize=14, y=1.02)

#     if save:
#         save_plot('experiment_1')
#     plt.show()
