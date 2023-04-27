import time
import json

import pandas as pd

from federatedscope.core.configs.config import global_cfg
from federatedscope.cloud.common.utils import config2cmdargs, flatten_dict
from federatedscope.cloud.common.manager import Manager


class RoomManager(Manager):
    def __init__(self,
                 user,
                 organizer,
                 logger,
                 white_list=[],
                 black_list=[],
                 config=None):
        super(RoomManager, self).__init__([
            'idx', 'abstract', 'cfg', 'auth', 'log_file', 'port', 'pid',
            'cur_client', 'max_client'
        ],
                                          user=user,
                                          white_list=white_list,
                                          black_list=black_list)
        self.organizer = organizer
        self.logger = logger
        # TODO: replace test data with real data
        self.df_match = pd.DataFrame(
            data=[[34, 'tabular', 'dim_tb_itm', '0.93', ''],
                  [12, 'tabular', 'new_taobao', '0.79', ''],
                  [2, 'tabular', 'tb_ord_ent1', '0.68', ''],
                  [4, 'tabular', 'tb_ord_ent2', '0.65', ''],
                  [18, 'tabular', 'tb_ord_ent3', '0.64', '']],
            columns=['idx', 'domain', 'abstract', 'rate', 'status'])
        self.df_request = pd.DataFrame(
            data=[[1, 3, 'tabular', 'user_mark_result', '0.84', ''],
                  [2, 3, 'tabular', 's_user_data', '0.73', ''],
                  [3, 3, 'tabular', 'dim_tb_seller_maincate', '0.71', ''],
                  [4, 4, 'tabular', 'dim_tb_shop1', '0.62', ''],
                  [5, 4, 'tabular', 'dim_tb_shop2', '0.58', '']],
            columns=[
                'idx', 'room_idx', 'domain', 'abstract', 'rate', 'status'
            ])
        self.config = config

    def display(self, condition={}):
        # Sync and display lobby
        result = self.organizer.send_task('server.display_lobby', [self.auth])

        cnt = 0
        while (not result.ready()) and cnt < self.config.timeout:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        lobby = pd.read_json(result.get(timeout=1))

        if len(lobby) == 0:
            self.logger.info(
                'No room available now. Please create a new room.')
            return pd.DataFrame(data=[[None] * 9], columns=self.df.columns)
        else:
            oudated_room = []
            for i in range(len(self.df)):
                room = self.df.loc[i]
                if room['idx'] in lobby['idx'] and room['abstract'] in \
                        lobby['abstract']:
                    pass
                else:
                    oudated_room.append(i)
            self.df = self.df.drop(oudated_room)

            # Filter by condition
            if condition:
                # convert `str` to `dict`
                for key, value in condition.items():
                    lobby = lobby.loc[lobby[key] == value]

            self.logger.info(lobby)
            return lobby

    def discover(self, lobby_disc_input_idx, lobby_disc_input_domain,
                 lobby_disc_input_key, lobby_disc_input_column,
                 lobby_disc_input_extract):
        # TODO: implement this
        for i in range(2):
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
        return self.df_match

    def display_request(self):
        # TODO: implement this
        for i in range(2):
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
        return self.df_request

    def respond_request(self, lobby_req_idx):
        # TODO: implement this
        for i in range(2):
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
        self.df_request.loc[self.df_request['idx'] == lobby_req_idx,
                            'status'] \
            = 'Agreed'
        return self.df_request

    def send_request(self, lobby_cand_idx):
        # TODO: implement this
        for i in range(3):
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
        self.df_match.loc[self.df_match['idx'] == lobby_cand_idx, 'status'] \
            = 'Request sent'
        return self.df_match

    # def display_process(self):
    #     # TODO: implement this
    #     for i in range(1):
    #         self.logger.info('Waiting for response... (will re-try in 1s)')
    #         time.sleep(1)

    def create(self, data_upload, data_choose, scenario, domain, yaml,
               is_private, opts, password, optimizer, model, feat, pruning,
               min_lr, max_lr, min_wd, max_wd):
        if is_private:
            return
        yaml = yaml.name
        cfg = global_cfg.clone()
        if yaml.endswith('.yaml'):
            cfg.merge_from_file(yaml)
        else:
            self.logger.warning('The yaml file is none or invalid, ignored.')
        # TODO: merge yaml and other opts
        if data_choose:
            cfg.merge_from_list(['data.type', data_choose])
        self.add(cfg, opts, password)

    def add(self, yaml, opts='', password='123'):
        # TODO: add wandb related args
        opts = opts.split(' ')
        cfg = yaml
        if len(opts) > 1:
            cfg.merge_from_list(opts)

        cfg = config2cmdargs(flatten_dict(cfg))

        command = ''
        for i in cfg:
            value = f'{i}'.replace(' ', '')
            command += f' "{value}"'
        args = command[1:]
        result = self.organizer.send_task('server.add_room',
                                          [args, password, self.auth])
        cnt = 0
        while (not result.ready()) and cnt < self.config.timeout:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        msg = result.get(timeout=1)
        self.logger.info(msg)

    def authorize(self, idx, password):
        result = self.organizer.send_task('server.auth_room',
                                          [idx, password, self.auth])
        cnt = 0
        while (not result.ready()) and cnt < self.config.timeout:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        info = result.get(timeout=1)
        try:
            room = json.loads(s=info)
            self.logger.info(f'Get authorization of room {idx}.')
            if room['idx'] in self.df['idx']:
                self.df.loc[self.df['idx'] == room['idx']] = room
            else:
                self.df.loc[len(self.df)] = room
        except:
            self.logger.info(info)

    def send_authorize(self, idx, action):
        # TODO: implement this
        for i in range(3):
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
        self.logger.info(f'Request {idx} {action}.')

    def shutdown(self, idx=0):
        # Shut down room
        if idx == 0:
            idx = None
        result = self.organizer.send_task('server.shutdown', [idx, self.auth])
        cnt = 0
        while (not result.ready()) and cnt < self.config.timeout:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        self.logger.info(result.get(timeout=1))


if __name__ == '__main__':
    ...
    # import logging
    # from celery import Celery
    #
    # from federatedscope.organizer import cfg_client
    # organizer = Celery()
    # organizer.config_from_object(cfg_client)
    # logging.basicConfig(level=logging.INFO)
    #
    # rm = RoomManager('root', organizer, logging)
    #
    # # Test functions
    # rm.display_room()
    # rm.add(
    #     'scripts/distributed_scripts/distributed_configs'
    #     '/distributed_femnist_server.yaml',
    #     password=12345)
    # rm.display_room()
    # rm.authorize(1, 12345)
    # rm.authorize(1, 12345)
    # rm.display_room()
    # rm.shutdown(1)
    # rm.shutdown(2)
    # rm.shutdown()
