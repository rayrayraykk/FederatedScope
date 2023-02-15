from federatedscope.core.configs.config import global_cfg

from federatedscope.organizer.module.ssh import SSHManager
from federatedscope.organizer.module.manager import Manager
from federatedscope.organizer.utils import config2cmdargs, flatten_dict

from federatedscope.organizer.cfg_client import server_ip as SERVER_IP


class ECSManager(Manager):
    def __init__(self,
                 user,
                 logger,
                 room_manager,
                 white_list=[],
                 black_list=[]):
        super(ECSManager, self).__init__(['ip', 'user', 'password', 'manager'],
                                         user=user,
                                         white_list=white_list,
                                         black_list=black_list)
        self.logger = logger
        self.room_manager = room_manager

    def display(self, condition={}):
        if condition:
            for key, value in condition.items():
                df = self.df.loc[self.df[key] == value]
        else:
            df = self.df

        if len(df):
            self.logger.info(df)
        else:
            self.logger.info('No eligible ECS, please add ECS via `AddECS` '
                             'first!')

    def add(self, ip, user, password):
        if ip in self.df['ip']:
            raise ValueError(f"ECS `{ip}` already exists.")
        self.df.loc[len(self.df)] = {
            'ip': ip,
            'user': user,
            'password': password,
            'manager': SSHManager(ip, user, password)
        }
        self.logger.info(f"{ip} added.")

    def shutdown(self, ip):
        for i in range(len(self.df)):
            if ip == self.df.loc[i]['ip']:
                self.df.loc[i]['manager']._disconnect()
                break
        self.df.drop(i)

    def join(self, idx, ip, opts):
        lobby = self.room_manager.df
        try:
            room = lobby.loc[lobby['idx'] == ip].iloc[0]
        except:
            return f'Room {idx} not found or authorized.'

        try:
            manager = self.df.loc[self.df['ip'] == ip].iloc[0]
        except:
            return f'ECS {ip} not found.'

        cfg = global_cfg.clone()
        room_cfg = room['cfg']
        cfg.merge_from_list(room_cfg)

        # Merge other opts and convert to command string
        if len(opts):
            opts = opts.split(' ')
            cfg.merge_from_list(opts)

        # Convert necessary configurations
        cfg['distribute']['server_host'] = SERVER_IP
        cfg['distribute']['client_host'] = ip
        cfg['distribute']['role'] = 'client'
        cfg['distribute']['server_port'] = room['port']
        cfg['distribute']['client_port'] = self.find_free_port()

        cfg = config2cmdargs(flatten_dict(cfg))
        command = ''
        for i in cfg:
            value = f'{i}'.replace(' ', '')
            command += f' "{value}"'
        command = command[1:]
        pid = manager.launch_task(command)
        self.logger.info(f'{ip}({pid}) launched,')
