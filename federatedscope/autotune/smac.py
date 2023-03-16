import os
import logging
import numpy as np
import ConfigSpace as CS
from federatedscope.autotune.utils import eval_in_fs, log2wandb, \
    summarize_hpo_results
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.multi_objective.parego import ParEGO

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

trial_index = 0


def run_smac(cfg, scheduler, client_cfgs=None):
    config_space = scheduler._search_space
    init_configs = []
    perfs = []

    def optimization_function_wrapper(config):
        """
        Used as an evaluation function for SMAC optimizer.
        Args:
            config: configurations of FS run.
        Returns:
            Best results of server of specific FS run.
        """
        global trial_index

        budget = cfg.hpo.sha.budgets[-1]
        results = eval_in_fs(cfg, config, budget, client_cfgs, trial_index,
                             config_space)
        key1, key2 = cfg.hpo.metric.split('.')
        res = results[key1][key2]
        if cfg.hpo.scheduler == 'multi':
            new_res = {cfg.hpo.metric: res}
            for key, w in zip(cfg.hpo.multi_obj.key, cfg.hpo.multi_obj.weight):
                key1, key2 = key.split('.')
                new_res[key] = w * results[key1][key2]
            res = new_res
        config = dict(config)
        config['federate.total_round_num'] = budget
        init_configs.append(config)

        if isinstance(res, dict):
            perfs.append(sum(res.values()))
        else:
            perfs.append(res)

        logger.info(f'Evaluate the {len(perfs)-1}-th config '
                    f'{config}, and get performance {res}')
        if cfg.wandb.use:
            tmp_results = \
                summarize_hpo_results(init_configs,
                                      perfs,
                                      white_list=set(config_space.keys()),
                                      desc=cfg.hpo.larger_better,
                                      is_sorted=False)
            log2wandb(len(perfs) - 1, config, results, cfg, tmp_results)
        trial_index += 1

        if cfg.hpo.larger_better:
            return -res
        else:
            return res

    def summarize():
        results = summarize_hpo_results(init_configs,
                                        perfs,
                                        white_list=set(config_space.keys()),
                                        desc=cfg.hpo.larger_better,
                                        use_wandb=cfg.wandb.use)
        logger.info(
            "========================== HPO Final ==========================")
        logger.info("\n{}".format(results))
        results.to_csv(os.path.join(cfg.hpo.working_folder, 'results.csv'))
        logger.info("====================================================")
        logger.info(f'Winner config_id: {np.argmin(perfs)}')

        return perfs

    if cfg.hpo.sha.iter != 0:
        n_iterations = cfg.hpo.sha.iter
    else:
        n_iterations = -int(
            np.log(cfg.hpo.sha.budgets[0] / cfg.hpo.sha.budgets[-1]) /
            np.log(cfg.hpo.sha.elim_rate)) + 1

    if cfg.hpo.scheduler == 'multi':
        multi_obj_kwargs = {
            "multi_objectives": [cfg.hpo.metric] + cfg.hpo.multi_obj.key,
        }
    else:
        multi_obj_kwargs = {}

    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": n_iterations,
        "cs": config_space,
        "output_dir": cfg.hpo.working_folder,
        "deterministic": "true",
        "limit_resources": False,
        **multi_obj_kwargs
    })

    if cfg.hpo.scheduler.endswith('bo_gp'):
        smac = SMAC4BB(model_type='gp',
                       scenario=scenario,
                       tae_runner=optimization_function_wrapper)
    elif cfg.hpo.scheduler.endswith('bo_rf'):
        smac = SMAC4HPO(scenario=scenario,
                        tae_runner=optimization_function_wrapper)
    elif cfg.hpo.scheduler == 'multi':
        if cfg.hpo.multi_obj.algo == 'mean':
            smac = SMAC4BB(
                scenario=scenario,
                tae_runner=optimization_function_wrapper,
            )
        elif cfg.hpo.multi_obj.algo == 'parego':
            smac = SMAC4HPO(
                scenario=scenario,
                tae_runner=optimization_function_wrapper,
                multi_objective_algorithm=ParEGO,
                multi_objective_kwargs={
                    "rho": 0.05,
                },
            )
        else:
            raise ValueError(f'Unsupported `cfg.hpo.multi_obj.algo`'
                             f' {cfg.hpo.multi_obj.algo}')
    else:
        raise NotImplementedError
    try:
        smac.optimize()
    finally:
        smac.solver.incumbent
    summarize()
    return perfs
