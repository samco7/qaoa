from time import time
from classical_solvers import *
from helper_functions import *
from save_results import *
from tqdm import tqdm
from qiskit.providers.aer import AerSimulator


def experiment_0(graph_sizes, n_trials, save=True):
    weights_set = list(range(1, 11))
    gw_mean_ratios = []
    bmz_mean_ratios = []
    gw_mean_times = []
    bmz_mean_times = []
    n_vals = []
    progress = tqdm(total=len(graph_sizes)*n_trials)
    for n in np.random.choice(graph_sizes, size=len(graph_sizes), replace=False):
        n_vals.append(n)
        gw_ratios = []
        bmz_ratios = []
        gw_times = []
        bmz_times = []
        for i in range(n_trials):
            graph = random_graph(n, weights_set)
            exact_val = akmaxsat(graph)[1]
            start = time()
            gw_val = GW(graph)[1]
            gw_times.append(time() - start)
            start = time()
            bmz_val = BMZ(graph)[1]
            bmz_times.append(time() - start)
            gw_ratios.append(gw_val/exact_val)
            bmz_ratios.append(bmz_val/exact_val)
            progress.update(1)
        gw_mean_ratios.append(np.mean(gw_ratios))
        bmz_mean_ratios.append(np.mean(bmz_ratios))
        gw_mean_times.append(np.mean(gw_times))
        bmz_mean_times.append(np.mean(bmz_times))
    progress.close()

    n_vals = np.array(n_vals)
    gw_mean_ratios = np.array(gw_mean_ratios)
    bmz_mean_ratios = np.array(bmz_mean_ratios)
    gw_mean_times = np.array(gw_mean_times)
    bmz_mean_times = np.array(bmz_mean_times)
    index = np.argsort(n_vals)
    res = {'n_vals':n_vals, 'gw_mean_ratios':gw_mean_ratios, 'bmz_mean_ratios':bmz_mean_ratios, 'gw_mean_times':gw_mean_times, 'bmz_mean_times':bmz_mean_times, 'n_trials':n_trials}
    info = {'graph_sizes':graph_sizes, 'n_trials':n_trials, 'weights_set':weights_set}
    if save:
        save_result(res, 'experiment_0')
        save_info(info, 'experiment_0')
    return res


def plot_experiment_0(res, save=True):
    n_vals = res['n_vals']
    gw_mean_ratios, bmz_mean_ratios = res['gw_mean_ratios'], res['bmz_mean_ratios']
    gw_mean_times, bmz_mean_times = res['gw_mean_times'], res['bmz_mean_times']
    index = np.argsort(n_vals)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(n_vals[index], gw_mean_ratios[index], label='GW')
    ax[0].plot(n_vals[index], bmz_mean_ratios[index], label='BMZ')
    ax[0].legend()
    ax[0].set_xlabel('Graph nodes', fontsize=12)
    ax[0].set_ylabel('Mean accuracy', fontsize=12)

    ax[1].plot(n_vals[index], gw_mean_times[index], label='GW')
    ax[1].plot(n_vals[index], bmz_mean_times[index], label='BMZ')
    ax[1].legend()
    ax[1].set_xlabel('Graph nodes', fontsize=12)
    ax[1].set_ylabel('Mean computation time (s)', fontsize=12)
    plt.tight_layout()
    plt.suptitle(f'Relaxed solver performance', y=1.02, fontsize=14)

    if save:
        save_plot('experiment_0')
    plt.show()
