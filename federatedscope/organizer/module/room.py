import time

import pandas as pd

from federatedscope.core.configs.config import global_cfg
from federatedscope.organizer.utils import config2cmdargs, flatten_dict
from federatedscope.organizer.module.manager import Manager
from federatedscope.organizer.cfg_client import TIMEOUT


class RoomManager(Manager):
    def __init__(self, user, organizer, logger):
        super(RoomManager, self).__init__([
            'idx', 'abstract', 'cfg', 'auth', 'log_file', 'port', 'pid',
            'cur_client', 'max_client'
        ],
                                          user=user)
        self.organizer = organizer
        self.logger = logger

    def display(self, condition={}):
        # Sync and display lobby
        result = self.organizer.send_task('server.display_lobby', [self.auth])

        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        lobby = pd.read_json(result.get(timeout=1))
        if len(lobby) == 0:
            self.logger.info(
                'No task available now. Please create a new task.')
        else:
            for i in range(len(self.df)):
                room = self.df.loc[i]
                # TODO: update self.df
                print(room)

        # Filter by condition
        if condition:
            # convert `str` to `dict`
            for key, value in condition.items():
                lobby = lobby.loc[lobby[key] == value]

        self.logger.info(lobby)

    def add(self, yaml, opts='', password='123'):
        opts = opts.split(' ')
        cfg = global_cfg.clone()
        if yaml.endswith('.yaml'):
            cfg.merge_from_file(yaml)
        else:
            self.logger.warning('The yaml file is none or invalid, ignored.')
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
        while (not result.ready()) and cnt < TIMEOUT:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        msg = result.get(timeout=1)
        self.logger.info(msg)

    def authorize(self, idx, password):
        result = self.organizer.send_task('server.auth_room',
                                          [idx, password, self.auth])
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        info = result.get(timeout=1)
        if isinstance(info, pd.Series):
            room = info
            self.logger.info(room)
            # TODO: add to authorized Rooms
        else:
            self.logger.info(info)

    def shutdown(self, idx=None):
        # Shut down room
        result = self.organizer.send_task('server.shutdown', [idx, self.auth])
        cnt = 0
        while (not result.ready()) and cnt < TIMEOUT:
            self.logger.info('Waiting for response... (will re-try in 1s)')
            time.sleep(1)
            cnt += 1
        self.logger.info(result.get(timeout=1))


if __name__ == '__main__':
    import logging
    from celery import Celery

    from federatedscope.organizer import cfg_client
    organizer = Celery()
    organizer.config_from_object(cfg_client)
    logging.basicConfig(level=logging.INFO)

    rm = RoomManager('root', organizer, logging)

    # Test functions
    rm.display()
    rm.add('scripts/example_configs/femnist.yaml', password=12345)
    rm.display()
    rm.authorize(1, 12345)
    rm.shutdown(1)
    rm.shutdown()
