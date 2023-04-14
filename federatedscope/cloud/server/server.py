from celery import Celery

import federatedscope.cloud.server.config as cfg_server

from federatedscope.cloud.server.module.lobby import Lobby

# ---------------------------------------------------------------------- #
# Message related
# ---------------------------------------------------------------------- #
organizer = Celery('server',
                   broker='redis://localhost:6379/0',
                   backend='redis://localhost')
organizer.config_from_object(cfg_server)
lobby = Lobby()


# ---------------------------------------------------------------------- #
# Room related tasks
# ---------------------------------------------------------------------- #
@organizer.task
def display_lobby(auth):
    rtn_info = lobby.display(auth)
    print(rtn_info)
    return rtn_info


@organizer.task
def add_room(args, psw, auth):
    rtn_info = lobby.add(args, psw, auth)
    print(rtn_info)
    return rtn_info


@organizer.task
def auth_room(idx, password, auth):
    rtn_info = lobby.authorize(idx, password, auth)
    print(rtn_info)
    return rtn_info


@organizer.task
def shutdown(idx, auth):
    rtn_info = lobby.shutdown(idx, auth)
    print(rtn_info)
    return rtn_info
