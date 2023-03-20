import logging
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from federatedscope.autotune.utils import parse_search_space

logger = logging.getLogger(__name__)


def get_best_hyperpara(local_results_df, cfg):
    configs = []
    perfs = []

    if cfg.hpo.flora.aggregation == 'sgm':
        random_forest = RandomForestRegressor()
        x_train = np.vstack(
            [df.values[:, :-1] for df in local_results_df.values])
        y_train = np.vstack(
            [df.values[:, -1] for df in local_results_df.values])
        random_forest.fit(x_train, y_train)
    elif cfg.hpo.flora.aggregation == 'aplm' \
            or cfg.hpo.flora.aggregation == 'mplm':
        random_forest = {}
        for client, df in tqdm(local_results_df.items()):
            random_forest[client] = RandomForestRegressor()
            random_forest[client].fit(df.values[:, :-1], df.values[:, -1])
    else:
        raise NotImplementedError

    def eval_in_surrogate(config):
        input_x = [[config[x] for x in sorted(list(config.keys()))]]
        if cfg.hpo.flora.aggregation == 'sgm':
            model = random_forest
            perf = model.predict(input_x)
        elif cfg.hpo.flora.aggregation == 'aplm':
            preds = []
            for model in random_forest.values():
                preds.append(model.predict(input_x))
            perf = np.mean(preds)
        elif cfg.hpo.flora.aggregation == 'mplm':
            preds = []
            for model in random_forest.values():
                preds.append(model.predict(input_x))
            perf = np.max(preds)
        else:
            raise NotImplementedError
        configs.append(config)
        perfs.append(perf)
        logger.info(f'Evaluate the {len(perfs) - 1}-th config '
                    f'{config}, and get performance {perf}')
        return perf

    # Global Tune
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": cfg.hpo.flora.glob_iter,
        "output_dir": cfg.hpo.working_folder,
        "cs": parse_search_space(cfg.hpo.flora.ss),
        "deterministic": "true",
        "limit_resources": False,
    })

    if cfg.hpo.flora.global_tuner == 'bo_gp':
        smac = SMAC4BB(model_type='gp',
                       scenario=scenario,
                       tae_runner=eval_in_surrogate)
    elif cfg.hpo.flora.global_tuner == 'bo_rf':
        smac = SMAC4HPO(scenario=scenario, tae_runner=eval_in_surrogate)

    try:
        smac.optimize()
    finally:
        smac.solver.incumbent

    param = fix_type(dict(configs[np.argmin(perfs)]))
    logger.info(f'Find best hyper-parameters {param}.')

    return param


def fix_type(dic):
    int_key = ['train.local_update_steps', 'dataloader.batch_size']
    float_key = [
        'train.optimizer.lr', 'train.optimizer.weight_decay', 'model.dropout'
    ]
    str_key = []

    for key, value in dic.items():
        if key in float_key:
            dic[key] = float(value)
        elif key in str_key:
            dic[key] = str(value)
        elif key in int_key:
            dic[key] = int(value)
    return dic
