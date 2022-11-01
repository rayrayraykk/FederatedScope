import datetime
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.fed_runner import FedRunner

from fedhpob.benchmarks.base_benchmark import BaseBenchmark
from fedhpob.utils.util import disable_fs_logger
from fedhpob.utils.cost_model import merge_cfg


class RawBenchmark(BaseBenchmark):
    def __init__(self,
                 model,
                 dname,
                 algo,
                 rng=None,
                 cost_mode='estimated',
                 **kwargs):
        """
        Initialize a benchmark object.

        Args:
            self: write your description
            model: write your description
            dname: write your description
            algo: write your description
            rng: write your description
            cost_mode: write your description
        """
        super(RawBenchmark, self).__init__(model, dname, algo, cost_mode, rng,
                                           **kwargs)
        self.device = kwargs['device']

    def _run_fl(self, configuration, fidelity, key='val_avg_loss', seed=1):
        """
        Run Fed.

        Args:
            self: write your description
            configuration: write your description
            fidelity: write your description
            key: write your description
            seed: write your description
        """
        init_cfg = self.cfg.clone()
        disable_fs_logger(init_cfg, True)
        setup_seed(seed)
        modified_cfg = merge_cfg(init_cfg, configuration, fidelity)
        data, modified_cfg = get_data(modified_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        init_cfg.device = self.device
        if self.algo == 'opt':
            init_cfg.federate.share_local_model = False
            init_cfg.federate.online_aggr = False
            init_cfg.fedopt.use = True
            init_cfg.federate.method = 'FedOpt'
        init_cfg.freeze()
        runner = FedRunner(data=data,
                           server_class=get_server_cls(init_cfg),
                           client_class=get_client_cls(init_cfg),
                           config=init_cfg.clone())
        results = runner.run()
        # so that we could modify cfg in the next trial
        init_cfg.defrost()
        if 'server_global_eval' in results:
            return [results['server_global_eval'][key]]
        else:
            return [results['client_summarized_weighted_avg'][key]]

    def objective_function(self,
                           configuration,
                           fidelity=None,
                           key='val_avg_loss',
                           seed=1,
                           **kwargs):
        """
        Computes the objective function for the given configuration.

        Args:
            self: write your description
            configuration: write your description
            fidelity: write your description
            key: write your description
            seed: write your description
        """
        fidelity = self._init_fidelity(fidelity)
        self._check(configuration, fidelity)
        start_time = datetime.datetime.now()
        function_value = self._run_fl(configuration, fidelity, key, seed)[0]
        end_time = datetime.datetime.now()
        if self._cost(configuration, fidelity, **kwargs):
            cost = self._cost(configuration, fidelity, **kwargs)
        else:
            # TODO: use time from  FS monitor
            cost = end_time - start_time

        return {'function_value': function_value, 'cost': cost}

    def get_configuration_space(self):
        """
        Returns the configuration space for the current user.

        Args:
            self: write your description
        """
        return []

    def get_fidelity_space(self):
        """
        Returns the FIdelity space for the QuantFigure

        Args:
            self: write your description
        """
        return []

    def get_meta_info(self):
        """
        Return meta information about the dataset

        Args:
            self: write your description
        """
        return {
            'model': self.model,
            'dname': self.dname,
            'configuration_space': self.configuration_space,
            'fidelity_space': self.fidelity_space
        }
