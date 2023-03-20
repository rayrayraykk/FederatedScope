import copy
import logging
import numpy as np

from torch.utils.data import Subset
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from federatedscope.core.message import Message
from federatedscope.core.workers import Client
from federatedscope.autotune.utils import parse_search_space
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

logger = logging.getLogger(__name__)


class FLoRAClient(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):

        super(FLoRAClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, is_unseen_client, *args, **kwargs)

        subdata = copy.deepcopy(data)
        dataset = getattr(subdata, 'train_data')
        if dataset is not None:
            index = np.random.permutation(np.arange(len(dataset)))
            sub_size = int(self._cfg.hpo.flora.sample_loc_data * len(dataset))
            sub_dataset = Subset(dataset, index[:sub_size])
            setattr(subdata, 'train_data', sub_dataset)
        subdata.setup(self._cfg, force_init=True)

        self.loc_trainer = get_trainer(model=model,
                                       data=subdata,
                                       device=device,
                                       config=self._cfg,
                                       is_attacker=self.is_attacker,
                                       monitor=self._monitor)

        self.register_handlers('local_tune',
                               self.callback_funcs_for_local_tune,
                               ['local_results'])
        self.register_handlers('hyperparams',
                               self.callback_funcs_for_hyperparams,
                               ['ready_for_fl'])

    def _apply_hyperparams(self, hyperparams):
        """Apply the given hyperparameters
        Arguments:
            hyperparams (dict): keys are hyperparameter names \
                and values are specific choices.
        """
        cmd_args = []
        for k, v in hyperparams.items():
            cmd_args.append(k)
            cmd_args.append(v)

        self._cfg.defrost()
        self._cfg.merge_from_list(cmd_args, check_cfg=False)
        self._cfg.freeze(inform=False, check_cfg=False)
        self.trainer.cfg = self._cfg

    def callback_funcs_for_local_tune(self, message: Message):
        sender = message.sender
        self.state = message.state
        self.init_model_para = message.content[0]
        self.local_tune_res = []
        self.num_iter = 0

        # Local Tune
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": self._cfg.hpo.flora.loc_iter,
            "output_dir": self._cfg.hpo.working_folder,
            "cs": parse_search_space(self._cfg.hpo.flora.ss),
            "deterministic": "true",
            "limit_resources": False,
        })

        if self._cfg.hpo.flora.local_tuner == 'bo_gp':
            smac = SMAC4BB(model_type='gp',
                           scenario=scenario,
                           tae_runner=self._local_train)
        elif self._cfg.hpo.flora.local_tuner == 'bo_rf':
            smac = SMAC4HPO(scenario=scenario, tae_runner=self._local_train)

        try:
            smac.optimize()
        finally:
            smac.solver.incumbent

        self.comm_manager.send(
            Message(msg_type='local_results',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=self.local_tune_res))

    def callback_funcs_for_hyperparams(self, message: Message):
        sender, hyperparams = message.sender, message.content
        self._apply_hyperparams(hyperparams)
        self.comm_manager.send(
            Message(msg_type='ready_for_fl',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state))

    def _local_train(self, hyperparams):
        self._apply_hyperparams(hyperparams)
        self.loc_trainer.update(self.init_model_para)
        for i in range(self._cfg.hpo.flora.loc_epoch):
            self.loc_trainer.train()
            logger.info(f'\tClient #{self.ID} local tune @iter '
                        f'{self.num_iter} @Epoch {i}.')
        eval_metrics = self.loc_trainer.evaluate(target_data_split_name='val')
        res = eval_metrics['val_avg_loss']
        logger.info(f'\tClient #{self.ID} local tune @iter '
                    f'{self.num_iter} results: {res}.')
        self.local_tune_res.append((hyperparams, res))
        self.num_iter += 1
        return res
