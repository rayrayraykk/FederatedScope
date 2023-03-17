import os
import logging
from itertools import product

import yaml

import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp
import torch

from federatedscope.core.message import Message
from federatedscope.core.workers import Server

logger = logging.getLogger(__name__)


class FedExServer(Server):
    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        """
        Use this trigger as local tunner before broadcast
        """
        self.trigger_train_func = trigger_train_func
        self.kwargs_for_trigger_train_func = kwargs_for_trigger_train_func

        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = [self.model.state_dict()]

        self.comm_manager.send(
            Message(msg_type='local_tune',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    state=self.state,
                    content=model_para))
