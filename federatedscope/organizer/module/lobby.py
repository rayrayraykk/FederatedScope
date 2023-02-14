import os
import redis
import pickle
import signal
import subprocess
from datetime import datetime

from federatedscope.organizer.module.manager import Manager
from federatedscope.organizer.utils import args2yaml, config2cmdargs, \
    flatten_dict


class Lobby(Manager):
    def __init__(self, host='localhost', port=6379, db=0):
        super(Lobby, self).__init__([
            'idx', 'abstract', 'cfg', 'password', 'auth', 'log_file', 'port',
            'pid', 'cur_client', 'max_client'
        ],
                                    user='root')
        self.database = redis.StrictRedis(host=host, port=port, db=db)
        self._save('lobby', self.df)
        self._save('auth', self.auth)

    def _save(self, key, value):
        """
        Save object to Redis via pickle.
        """
        pickled_object = pickle.dumps(value)
        self.database.set(key, pickled_object)

    def _load(self, key):
        """
        Load object from Redis via pickle.
        """
        try:
            value = pickle.loads(self.database.get(key))
        except TypeError:
            value = None
        return value

    def _refresh_lobby(self):
        """
        Refresh room status and remove finished or dead room.
        """
        dead_pids = []
        lobby = self._load('lobby')
        for i in range(len(lobby)):
            pid = lobby.loc[i]['pid']
            if not self.get_cmd_from_pid(pid):
                dead_pids.append(i)
        if dead_pids:
            lobby = lobby.drop(dead_pids)
            self._save('lobby', lobby)

    def _check_user(self, user, is_root=False):
        """
        Check the validity of the user. If white list is enabled, user must
        be in the white list. If white list is not enabled, user must not be in
        the black list.
        """
        auth = self._load('auth')

        if is_root:
            return user == auth['owner']

        if len(auth['white_list']) > 0:
            if user not in auth['white_list']:
                return False
        else:
            if user in auth['black_list']:
                return False
        return True

    def add(self, args, password, auth):
        """
        Create FS server session and store args in Redis.
        """
        if not self._check_user(auth['owner']):
            return 'You are not permitted！'
        self._refresh_lobby()
        lobby = self._load('lobby')
        # Update room args in Redis
        if list(lobby['idx']):
            new_room_idx = self.get_missing_number(list(lobby['idx']))
        else:
            new_room_idx = 1
        cfg = args2yaml(args)

        # Update cfg
        cfg.distribute.server_port = self.find_free_port()

        cmd_cfg = config2cmdargs(flatten_dict(cfg))

        room = {
            'idx': new_room_idx,
            'abstract': f'{cfg.data.type} {cfg.model.type}',  # TODO: prettify
            'cfg': cmd_cfg,
            'password': password,
            'auth': auth,
            'log_file': os.path.join(
                'logs',
                str(datetime.now().strftime('log_%Y%m%d%H%M%S')) + '.out'),
            'port': cfg.distribute.server_port,
            'pid': None,  # default, to be updated after launch
            'cur_client': 0,
            'max_client': cfg.federate.client_num
        }

        # Launch FS
        input_args = [str(x) for x in cmd_cfg]
        cmd = ['python', '../../federatedscope/main.py'] + input_args
        log = open(room['log_file'], 'a')
        p = subprocess.Popen(cmd, stdout=log, stderr=log)
        # Update pid
        room['pid'] = p.pid

        # Update lobby
        lobby.loc[len(lobby)] = room
        self._save('lobby', lobby)
        return f"The room was created successfully with Room {room['idx']}."

    def display(self, auth):
        """
        Display FS lobby.
        """
        if not self._check_user(auth['owner']):
            return 'You are not permitted！'

        self._refresh_lobby()
        mask_key = ['cfg', 'password', 'auth', 'pid']  # Important information
        lobby = self._load('lobby')
        for mask in mask_key:
            del lobby[mask]
        return lobby.to_json()

    def authorize(self, idx, password, auth):
        """
        Auth and send key of certain room back.
        """
        if not self._check_user(auth['owner']):
            return 'You are not permitted！'

        self._refresh_lobby()
        lobby = self._load('lobby')
        if idx in list(lobby['idx']):
            # Check the validity of the room
            room = lobby.loc[lobby['idx'] == idx].loc[0]
            if room['cur_client'] < room['max_client']:
                # Joinable, check auth and password
                room_auth, user = room['auth'], auth['owner']
                if len(room_auth['white_list']) > 0:
                    if user not in room_auth['white_list']:
                        return f'You are not in the white list of room {idx}'
                else:
                    if user in room_auth['black_list']:
                        return f'You are in the black list of room {idx}'

                # Check password
                if password != room['password']:
                    return 'Wrong Password!'
                else:
                    return room.to_json()
            else:
                # Full
                return f'Room {idx} is full'
        else:
            # Room does not exist
            return f'Room {idx} does not exist'

    def shutdown(self, idx, auth):
        """
        Shut down all or a certain room
        """
        if idx:
            if not self._check_user(auth['owner']):
                return 'You are not permitted！'
            self._refresh_lobby()
            lobby = self._load('lobby')

            if len(lobby.loc[lobby['idx'] == idx]):
                room = lobby.loc[lobby['idx'] == idx].iloc[0]
                room_auth, user = room['auth'], auth['owner']
                if room_auth['owner'] == user:
                    try:
                        os.kill(room['pid'], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    return f'Shut down room {idx} successfully.'
                else:
                    return 'You are not permitted'
            else:
                return 'Non-existent room ID.'
        else:
            if not self._check_user(auth['owner'], is_root=True):
                return 'You are not permitted！'
            else:
                self._refresh_lobby()
                lobby = self._load('lobby')
                for p in lobby['pid']:
                    try:
                        # os.kill(p, signal.SIGTERM)
                        subprocess.Popen(['kill', '-9', f'{p}'],
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                    except ProcessLookupError as error:
                        pass
                return 'Shut down all rooms successfully.'
