import copy
import logging

from federatedscope.core.data import ClientData
from federatedscope.core.workers import Server, Client
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

logger = logging.getLogger(__name__)


def change_label(data, label):
    new_dataset = []
    for (x, y), new_y in zip(data, label):
        new_dataset.append((x, new_y))
    return new_dataset


class PADServer(Server):
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
        super(PADServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        self.init_models = [copy.deepcopy(x) for x in self.models]
        # Client 1 is the master to match others
        self.candidates_id = 1
        self.sample_client_num = 2
        self._total_round_num = self._cfg.matching.round * self.client_num

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        if msg_type == 'model_para' and \
                self.state % self._cfg.matching.round == 0:
            self.candidates_id += 1
            self.unseen_clients_id = [x for x in range(2, self.client_num + 1)]
            if self.candidates_id in self.unseen_clients_id:
                self.unseen_clients_id.remove(self.candidates_id)
            else:
                self.save_best_results()
                self.terminate(msg_type='finish')
                return

            for model, init_model in zip(self.models, self.init_models):
                model.load_state_dict(init_model.state_dict())
        super(PADServer,
              self).broadcast_model_para(msg_type, sample_client_num,
                                         filter_unseen_clients)


class PADClient(Client):
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
        super(PADClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, is_unseen_client, *args, **kwargs)
        # Modify data labels
        label = 1 if self.ID == 1 else 0
        new_data = {'client_cfg': self._cfg}
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(data, split):
                split_data = getattr(data, split)
                if split_data is not None:
                    new_data[split.split('_')[0]] = \
                        change_label(split_data, [label] * len(split_data))
        data = ClientData(**new_data)
        self.trainer = get_trainer(model=model,
                                   data=data,
                                   device=device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker,
                                   monitor=self._monitor)
