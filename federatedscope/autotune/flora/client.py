import logging
import json
import copy
import numpy as np

from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from federatedscope.core.message import Message
from federatedscope.core.workers import Client
from federatedscope.autotune.utils import parse_search_space

logger = logging.getLogger(__name__)


class FLoRAClient(Client):
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
        self.init_model_para = message.content
        self.local_tune_res = []

        # Local Tune
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": self._cfg.hpo.flora.iter,
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
                    receiver=sender,
                    state=self.state,
                    content=self.local_tune_res))

    def _local_train(self, config):
        res = ...

        self.local_tune_res.append((config, res))
