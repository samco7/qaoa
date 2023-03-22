import pickle
import os
from os.path import exists
from datetime import datetime
import pytz
import sys
from helper_functions import *


def save_result(res, experiment_name):
    """
        Saves the results of an experiment as a pickle file.

        Parameters:
            res (any): result to be saved
            experiment_name (str): type of experiment the results correspond to (e.g. clustering
                or density).
    """
    # make a filename and directory corresponding to the date
    date = datetime.now(pytz.timezone('US/Eastern'))
    add_zero = lambda x: ('0' + str(x))[-2:]
    day_path = f'./../../results/{experiment_name}/'
    day_path += f'{date.year}-{add_zero(date.month)}-{add_zero(date.day)}'
    day_path += '/'
    if not exists(day_path):
        os.mkdir(day_path)
    filename = f'{date.year}-{add_zero(date.month)}-{add_zero(date.day)}_results'
    check = filename
    counter = 1
    while True:
        if exists(day_path + check + '.pickle'):
            check = filename
            check += '_' + str(counter)
            counter += 1
            continue
        break

    # save the results as a .pickle file
    with open(day_path + check + '.pickle', 'wb') as f:
        pickle.dump(res, f)


def save_plot(experiment_name, filename_override=None):
    """
        Saves the results of an experiment as a pickle file.

        Parameters:
            experiment_name (str): type of experiment the results correspond to (e.g. clustering
                or density).
    """
    # make a filename and directory corresponding to the date
    date = datetime.now(pytz.timezone('US/Eastern'))
    add_zero = lambda x: ('0' + str(x))[-2:]
    day_path = f'./../../plots/{experiment_name}/'
    day_path += f'{date.year}-{add_zero(date.month)}-{add_zero(date.day)}'
    day_path += '/'
    if not exists(day_path):
        os.mkdir(day_path)
    filename = f'{date.year}-{add_zero(date.month)}-{add_zero(date.day)}'
    check = filename
    counter = 1
    while True:
        if exists(day_path + check + '.png'):
            check = filename
            check += '_' + str(counter)
            counter += 1
            continue
        break

    if filename_override is not None:
        check = filename_override

    # save the plot (as a png, with double resolution)
    plt.savefig(day_path + check + '.png', facecolor='white', bbox_inches='tight', dpi=200)


def load_result(path):
    """
        Loads the results of an experiment that has been saved as a pickle file.

        Parameters:
            path (str): a filepath to a pickle file containing the results of an experiment.
                Meant to retrieve a result saved using the save_result function above.

        Returns:
            loaded_res (dict): the dictionary of the form saved in the save_result function above
                that we have retrieved.
    """
    with open(path, 'rb') as f:
        loaded_res = pickle.load(f)
    return loaded_res
