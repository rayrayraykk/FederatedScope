import logging

from federatedscope.core.message import Message
from federatedscope.core.workers import Server
from federatedscope.autotune.utils import summarize_hpo_results
from federatedscope.autotune.flora.utils import get_best_hyperpara

logger = logging.getLogger(__name__)


class FLoRAServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(FLoRAServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        self.register_handlers('local_results',
                               self.callback_funcs_for_local_results,
                               ['hyperparams'])
        self.register_handlers('ready_for_fl',
                               self.callback_funcs_for_ready_for_fl,
                               ['model_para'])

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

        # Cache for local_results
        self.msg_buffer['local_results'] = {}
        self.msg_buffer['ready_for_fl'] = {}

        self.comm_manager.send(
            Message(msg_type='local_tune',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    state=self.state,
                    content=model_para))

    def callback_funcs_for_local_results(self, message: Message):
        sender, local_tune_res = message.sender, message.content
        self.msg_buffer['local_results'][sender] = local_tune_res

        if len(self.msg_buffer['local_results']) == self._client_num:

            local_results_df = {}
            for client, client_local_results in \
                    self.msg_buffer['local_results'].items():
                configs, perfs = [], []
                for config, perf in client_local_results:
                    configs.append(config)
                    perfs.append(perf)
                # Convert to DataFrame
                local_results_df[client] = \
                    summarize_hpo_results(configs, perfs)

            best_hyperparams = \
                get_best_hyperpara(local_results_df,
                                   self._cfg.hpo.flora.aggregation)
            self.comm_manager.send(
                Message(msg_type='hyperparams',
                        sender=self.ID,
                        receiver=list(
                            self.comm_manager.get_neighbors().keys()),
                        state=self.state,
                        content=best_hyperparams))

    def callback_funcs_for_ready_for_fl(self, message: Message):
        sender = message.sender
        self.msg_buffer['ready_for_fl'][sender] = True

        if len(self.msg_buffer['ready_for_fl']) == self._client_num:
            del self.msg_buffer['ready_for_fl']
            self.trigger_train_func(**self.kwargs_for_trigger_train_func)
