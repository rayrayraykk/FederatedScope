# TODO: remove update; status update

import os
import random
import pandas as pd

from enum import Enum
from os.path import join as osp
from fabric import ThreadingGroup


class Status(Enum):
    RUNNING = 'RUNNING'
    NOT_RUNNING = 'NOT_RUNNING'
    KILLED = 'KILLED'
    FAILED = 'FAILED'
    UNKNOWN = 'UNKNOWN'


class FLManager(object):
    def __init__(self, hosts, user, password, port, logger, config):
        # Connections kwargs
        self.hosts = list(hosts)
        self.user = user
        self.password = password
        self.port = port

        # NOTE: May hang when multiple machines run in parallel
        self.runner = ThreadingGroup(
            *self.hosts,
            user=self.user,
            port=self.port,
            connect_kwargs={"password": self.password})

        # Initialization kwargs
        self.logger = logger
        self.config = config
        self.conda = osp(config.conda.path, 'bin', 'conda')
        self.env = osp(self.config.conda.path, 'envs', config.conda.env_name)
        self.python = osp(self.env, 'bin', 'python')
        self.pip = osp(self.env, 'bin', 'pip')

        self.setup()

        # FL task kwargs
        self.log_dir = "/tmp_fs_cloud"
        self.pid_dir = "/tmp_fs_cloud"
        self.index = 0  # Load balancing strategy

        # DataFrame to gradio
        self.df = pd.DataFrame(columns=["host", "command", "status"])

    def _round_robin(self):  # Load balancing strategy
        host = self.hosts[self.index]  # get the current host
        self.index = (self.index + 1) % len(self.hosts)
        return host

    def _random_choice(self):  # Load balancing strategy
        return random.choice(self.hosts)

    def _least_load(self):  # Load balancing strategy
        load_averages = {}
        for connection in self.runner:
            result = connection.run("cat /proc/loadavg", hide=True)
            load_averages[connection] = float(result.stdout.split()[0])
        return min(load_averages, key=load_averages.get)

    def _check_and_install(self, hosts, check_cmd, install_cmd, update_cmd=""):
        print(hosts, *hosts)
        group = ThreadingGroup(*hosts,
                               user=self.user,
                               port=self.port,
                               connect_kwargs={"password": self.password})

        print(check_cmd)
        try:
            res = group.run(check_cmd, warn=True)
        except Exception as error:
            print(error)
            return
        print('done', check_cmd)
        failed_groud = []
        for connection, result in res.items():
            if result.ok:
                self.logger.info(f"{connection.host} already has installed"
                                 f" ('{check_cmd}').")
                print(f"{connection.host} already has installed"
                      f" ('{check_cmd}').")
            else:
                failed_groud.append(connection)
        if len(failed_groud):
            tg = ThreadingGroup.from_connections(failed_groud)
            tg.run(install_cmd)
            self.logger.info(f"Successfully installed with '{install_cmd}'.")
            print(f"Successfully installed with '{install_cmd}'.")

        group.close()

        if len(update_cmd):
            try:
                group.run(update_cmd, warn=True)
            except Exception as error:
                print(error)
                return

    def _install_conda(self, hosts):
        check_cmd = f"{self.conda} --version"
        install_cmd = f"wget https://repo.anaconda.com/miniconda/Miniconda3" \
                      f"-py39_23.1.0-1-Linux-x86_64.sh; bash " \
                      f"Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -u -b -p" \
                      f" {self.config.conda.path}"
        self._check_and_install(hosts, check_cmd, install_cmd)

    def _install_python(self, hosts):
        check_cmd = f"{self.conda} env list | " \
                    f"grep {self.env}"

        install_cmd = f"{self.conda} " \
                      f"create -n {self.config.conda.env_name} " \
                      f"python={self.config.conda.python} -y"
        self._check_and_install(hosts, check_cmd, install_cmd)

    def _pip_install_package(self,
                             hosts,
                             package,
                             opt="",
                             repo_url="",
                             git_tag="",
                             package_version="",
                             pip_index="",
                             is_force_update=False):
        if not is_force_update:
            check_cmd = f"{self.conda} list " \
                        f"-n {self.config.conda.env_name} " \
                        f"| grep {package}"
        install_cmd = f"{self.pip} install {opt} " \
                      f"git+{repo_url}@{git_tag}" \
                      f"#egg={package}{package_version}" \
            if repo_url else \
            f"{self.pip} install {opt} {package}{package_version}"

        if pip_index:
            install_cmd += f" -i {pip_index}"

        self._check_and_install(hosts, check_cmd, install_cmd)

    def _install_backend(self, hosts):
        if self.config.backend.type != 'torch':
            raise NotImplementedError(f"{self.config.backend.type}")

        check_cmd = f"{self.conda} list " \
                    f"-n {self.config.conda.env_name} pytorch | " \
                    f"grep {self.config.backend.pytorch}"
        install_cmd = f"{self.conda} install -y " \
                      f"-n {self.config.conda.env_name} " \
                      f"pytorch={self.config.backend.pytorch} " \
                      f"torchvision={self.config.backend.torchvision} " \
                      f"torchaudio={self.config.backend.torchaudio} " \
                      f"torchtext={self.config.backend.torchtext} " \
                      f"cudatoolkit={self.config.backend.cudatoolkit} " \
                      f"-c pytorch -c conda-forge"
        self._check_and_install(hosts, check_cmd, install_cmd)

    # ---------------------------------------------------------------------- #
    # Not used method
    # ---------------------------------------------------------------------- #
    def _conda_install_package(self, hosts, package, extra_cmd=""):
        check_cmd = f"{self.conda} list " \
                    f"-n {self.config.conda.env_name} " \
                    f"| grep {package}"
        install_cmd = f"{self.conda} install -y {package} {extra_cmd}"

        self._check_and_install(hosts, check_cmd, install_cmd)

    # ---------------------------------------------------------------------- #
    # Not used method
    # ---------------------------------------------------------------------- #
    def _download_repo(self, hosts, repo_url, branch, path):
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        check_cmd = f"test -d {osp(path, repo_name, '.github')} || " \
                    f"test -d {osp(path, repo_name, 'README.md')}"
        install_cmd = f"rm -rf {osp(path, repo_name)}; " \
                      f"git clone {repo_url} {osp(path, repo_name)}"
        update_cmd = f"cd {osp(path, repo_name)}; " \
                     f"git checkout {branch}; " \
                     f"timeout 10 git fetch; " \
                     f"timeout 10 git pull"
        self._check_and_install(hosts, check_cmd, install_cmd, update_cmd)

    # ---------------------------------------------------------------------- #
    # Not used method
    # ---------------------------------------------------------------------- #
    def _run_python(self, command):
        result = self.runner.run(f"{self.python} -c "
                                 f"'{command}'")
        self.logger.info(f"{self.runner.host} {result.stdout.strip()}")

    def run(self, command):
        log_file = os.path.join(self.log_dir,
                                command.replace(" ", "_") + ".log")
        err_file = os.path.join(self.log_dir,
                                command.replace(" ", "_") + ".err")
        pid_file = os.path.join(self.pid_dir,
                                command.replace(" ", "_") + ".pid")
        nohup_command = f"nohup {command} > {log_file} 2> {err_file} & echo " \
                        f"$! > {pid_file}"

        # Load balancing
        if self.config.runner.balance == 'round_robin':
            selector = self._round_robin
        elif self.config.runner.balance == 'least_load':
            selector = self._least_load
        else:
            selector = self._random_choice

        result = self.runner.run(nohup_command, pty=False, selector=selector)
        self.logger.info(f"{self.runner.host} {result.stdout.strip()}")

    def kill(self, command):
        # kill a command by its pid file
        pid_file = os.path.join(self.pid_dir,
                                command.replace(" ", "_") + ".pid")
        kill_command = f"kill -9 $(cat {pid_file})"
        self.runner.run(kill_command)

    def status(self, command):
        # TODO: to be fixed
        pid_file = os.path.join(self.pid_dir,
                                command.replace(" ", "_") + ".pid")
        status_command = f"ps -p $(cat {pid_file})"
        result = self.runner.run(status_command)

    def setup(self, host=None):
        if host is None:
            hosts = self.hosts
        else:
            hosts = list(host)

        self._install_conda(hosts=hosts)
        self._install_python(hosts=hosts)
        self._install_backend(hosts=hosts)
        self._pip_install_package(hosts=hosts,
                                  package='federatedscope',
                                  repo_url=self.config.fs.repo_url,
                                  git_tag=self.config.fs.branch,
                                  package_version="[cloud]",
                                  is_force_update=True)

    def add_host(self, host):  # TODO: if host is a list
        if host in self.hosts:
            return

        self.setup(host)
        self.hosts.append(host)

        self.runner.close()
        self.runner = ThreadingGroup(
            *self.hosts,
            user=self.user,
            port=self.port,
            connect_kwargs={"password": self.password})

    def delete_host(self, host):  # TODO: if host is a list
        if host in self.hosts:
            self.hosts.remove(host)
            self.runner.close()
            self.runner = ThreadingGroup(
                *self.hosts,
                user=self.user,
                port=self.port,
                connect_kwargs={"password": self.password})

    def update_df(self, host, command, status):
        filtered_df = self.df.loc[(self.df["host"] == host)
                                  & (self.df["command"] == command)]
        if not filtered_df.empty:
            index = filtered_df.index[0]
            self.df.loc[index, "status"] = status
        else:
            new_df = pd.DataFrame([[host, command, status]],
                                  columns=["host", "command", "status"])
            self.df = self.df.append(new_df).reset_index(drop=True)


if __name__ == '__main__':
    import argparse
    import logging
    from federatedscope.cloud.client.config import cloud_cfg

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('hosts',
                        type=str,
                        help='a list of hosts separated by commas')
    parser.add_argument('username',
                        type=str,
                        help='the username for logging in')
    parser.add_argument('password',
                        type=str,
                        help='the password for logging in')
    args = parser.parse_args()

    hosts = args.hosts.split(',')
    username = args.username
    password = args.password

    manager = FLManager(hosts=hosts,
                        user=username,
                        password=password,
                        port=22,
                        logger=logger,
                        config=cloud_cfg)

    print(manager.df)
