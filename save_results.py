import pickle
import os
from os.path import exists
from datetime import datetime
import pytz

def save_result(res):
    # make a filename and directory corresponding to the date
    date = datetime.now(pytz.timezone('US/Eastern'))
    add_zero = lambda x: ('0' + str(x))[-2:]
    day_path = './results/'
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

def load_result(path):
    with open(path, 'rb') as f:
        loaded_res = pickle.load(f)
    return loaded_res
