# Time Consumption

import os
import numpy as np
from datetime import *

FILE = ['out_femnist_' + x for x in ['0.2', '0.4', '0.6', '0.8', '1.0']]

all_tol = []
all_eval = []
all_p = []

for p in FILE:
    os.chdir(os.path.join('FederatedScope', p))

    trail_names = os.listdir()

    tol_list = []
    eval_list = []

    for dname in trail_names:
        with open(os.path.join(dname, 'exp_print.log'), 'r') as f:
            F = f.readlines()
            start_time = datetime.strptime(F[0][:19], '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(F[-1][:19], '%Y-%m-%d %H:%M:%S')
            tol = end_time - start_time
            cnt_eval = start_time - start_time
            for line in F:
                if 'Starting evaluation' in line:
                    eval_start_time = datetime.strptime(line[:19],
                                                        '%Y-%m-%d %H:%M:%S')
                if 'Results_weighted_avg' in line:
                    eval_end_time = datetime.strptime(line[:19],
                                                      '%Y-%m-%d %H:%M:%S')
                    eval_time = eval_end_time - eval_start_time
                    cnt_eval = cnt_eval + eval_time
            # print(tol, cnt_eval)
            tol_list.append(tol)
            eval_list.append(cnt_eval)
    all_tol.append(np.mean(tol_list))
    all_eval.append(np.mean(eval_list))
    all_p.append(np.mean(eval_list) / np.mean(tol_list))
    print(f'finish {p}')
    os.chdir('../..')

for i, j in zip(all_tol,all_eval):
    print(i, j) 
print(all_p)
print(np.mean(all_tol), np.mean(all_eval), (np.mean(all_eval)/ np.mean(all_tol)))